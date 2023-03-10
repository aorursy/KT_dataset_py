import timeit

import pandas as pd

import numpy as np

import nltk

from collections import Counter

from nltk.corpus import stopwords

from sklearn.metrics import log_loss

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from scipy.optimize import minimize

stops = set(stopwords.words("english"))

import xgboost as xgb

from sklearn.cross_validation import train_test_split

import multiprocessing

import difflib



train = pd.read_csv('../input/train.csv')[:10000]

test = pd.read_csv('../input/test.csv')[:10000]



tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

#cvect = CountVectorizer(stop_words='english', ngram_range=(1, 1))



tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)

tfidf.fit_transform(tfidf_txt)

#cvect.fit_transform(tfidf_txt)



def diff_ratios(st1, st2):

    seq = difflib.SequenceMatcher()

    seq.set_seqs(str(st1).lower(), str(st2).lower())

    return seq.ratio()



def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R



def get_features(df_features):

    print('nouns...')

    df_features['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

    df_features['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])

    df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long

    print('lengths...')

    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))

    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))

    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))

    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split()))

    print('india/indian')

    df_features['question1_india'] = df_features.question1.map(lambda x: 'india' in str(x))

    df_features['question2_india'] = df_features.question2.map(lambda x: 'india' in str(x))

    df_features['indiacator'] = df_features['question1_india'] + df_features['question2_india']

    print('difflib...')

    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long

    print('word match...')

    df_features['z_word_match'] = df_features.apply(word_match_share, axis=1, raw=True)

    print('tfidf...')

    df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))

    df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))

    df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(tfidf.transform([str(x)]).data))

    df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(tfidf.transform([str(x)]).data))

    df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(tfidf.transform([str(x)]).data))

    df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(tfidf.transform([str(x)]).data))

    

    print("df_features", df_features.shape)

    train_orig =  pd.read_csv('../input/train.csv')

    #test_orig =  pd.read_csv('../input/test.csv', header=0)

    

    # feature using frequency

    tic0=timeit.default_timer()

    df1 = df_features[['question1']].copy()

    df2 = df_features[['question2']].copy()

    #df1_test = df_features[['question1']].copy()

    #df2_test = df_features[['question2']].copy()



    df2.rename(columns = {'question2':'question1'},inplace=True)

    #df2_test.rename(columns = {'question2':'question1'},inplace=True)



    train_questions = df1.append(df2)

    #train_questions = train_questions.append(df1_test)

    #train_questions = train_questions.append(df2_test)

    #train_questions.drop_duplicates(subset = ['qid1'],inplace=True)

    train_questions.drop_duplicates(subset = ['question1'],inplace=True)



    train_questions.reset_index(inplace=True,drop=True)

    questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()

    train_cp = train_orig.copy()

    #test_cp = test_orig.copy()

    train_cp.drop(['qid1','qid2'],axis=1,inplace=True)



    #test_cp['is_duplicate'] = -1

    #test_cp.rename(columns={'test_id':'id'},inplace=True)

    #comb = pd.concat([train_cp,test_cp])

    comb = train_cp.copy()

    comb = comb[:10000]

    comb['q1_hash'] = comb['question1'].map(questions_dict)

    comb['q2_hash'] = comb['question2'].map(questions_dict)



    q1_vc = comb.q1_hash.value_counts().to_dict()

    q2_vc = comb.q2_hash.value_counts().to_dict()



    def try_apply_dict(x,dict_to_apply):

        try:

            return dict_to_apply[x]

        except KeyError:

            return 0

    #map to frequency space

    comb['z_q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

    comb['z_q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

    

    train_comb = comb[['z_q1_freq','z_q2_freq']]

    #train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_hash','q2_hash','q1_freq','q2_freq','is_duplicate']]

    #test_comb = comb[comb['is_duplicate'] < 0][['id','q1_hash','q2_hash','q1_freq','q2_freq']]

    

    #df_features = df_features.append(train_comb, axis = 1) 

    

    df_features = pd.concat([df_features, train_comb], axis=1)

    print("df_features after concat with comb", df_features.shape)

                                                  

    return df_features.fillna(0.0)



train = get_features(train)

#train.to_csv('train.csv', index=False)



col = [c for c in train.columns if c[:1]=='z']



pos_train = train[train['is_duplicate'] == 1]

neg_train = train[train['is_duplicate'] == 0]

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

train = pd.concat([pos_train, neg_train])



x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.2, random_state=0)



params = {}

params["objective"] = "binary:logistic"

params['eval_metric'] = 'logloss'

params["eta"] = 0.02

params["subsample"] = 0.7

params["min_child_weight"] = 1

params["colsample_bytree"] = 0.7

params["max_depth"] = 4

params["silent"] = 1

params["seed"] = 1632



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100) #change to higher #s

print(log_loss(train.is_duplicate, bst.predict(xgb.DMatrix(train[col]))))



test = get_features(test)

#test.to_csv('test.csv', index=False)



sub = pd.DataFrame()

sub['test_id'] = test['test_id']

sub['is_duplicate'] = bst.predict(xgb.DMatrix(test[col]))



sub.to_csv('z08_submission_xgb_01.csv', index=False)



#print(train.head(2))

#print(test.head(2))
