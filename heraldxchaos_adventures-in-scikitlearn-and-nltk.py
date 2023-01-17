import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss



import xgboost as xgb

import multiprocessing

import difflib

from nltk.corpus import stopwords

from nltk.metrics import jaccard_distance



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv("../input/train.csv").fillna("")

df_test = pd.read_csv("../input/test.csv").fillna("")
stops = set(stopwords.words("english"))
def build_dict(sentences):

#    from collections import OrderedDict



    '''

    Build dictionary of train words

    Outputs: 

     - Dictionary of word --> word index

     - Dictionary of word --> word count freq

    '''

    print('Building dictionary..'),

    wordcount = dict()

    #For each worn in each sentence, cummulate frequency

    for ss in sentences:

        for w in ss:

            if w not in wordcount:

                wordcount[w] = 1

            else:

                wordcount[w] += 1

    

    worddict = dict()

    for idx, w in enumerate(sorted(wordcount.items(), key = lambda x: x[1], reverse=True)):

        worddict[w[0]] = idx+2  # leave 0 and 1 (UNK)



    return worddict, wordcount
def generate_sequence(sentences, dictionary):

    '''

    Convert tokenized text in sequences of integers

    '''

    seqs = [None] * len(sentences)

    for idx, ss in enumerate(sentences):

        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in ss]



    return seqs
def normalize(x):

    return x.lower().split()
questions = df_train['question1'].tolist() + df_train['question2'].tolist()
tok_questions = [normalize(s) for s in questions]

worddict, wordcount = build_dict(tok_questions)
print(np.sum(list(wordcount.values())), ' total words ', len(worddict), ' unique words')
def jc(x):

    return jaccard_distance(set(x['s_question1']),set(x['s_question2']))
def cosine_d(x):

    a = set(x['s_question1'])

    b = set(x['s_question2'])

    d = len(a)*len(b)

    if (d == 0):

        return 0

    else: 

        return len(a.intersection(b))/d
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))

tfidf.fit_transform(questions)
def diff_ratios(st1, st2):

    seq = difflib.SequenceMatcher()

    seq.set_seqs(str(st1).lower(), str(st2).lower())

    return seq.real_quick_ratio()



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

    

    print('jaccard...')

    df_features['s_question1'] = generate_sequence(df_features['question1'].apply(normalize),worddict)

    df_features['s_question2'] = generate_sequence(df_features['question2'].apply(normalize),worddict)

    df_features['z_jaccard'] = df_features.apply(jc,axis = 1)

    

    print('cosine....')

    df_features['z_cosine'] = df_features.apply(cosine_d,axis = 1)

    

    print('length....')

    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))

    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))

    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))

    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split())) 

    

    print('difflib...')

    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long

    

    print('word match...')

    df_features['z_word_match'] = df_features.apply(word_match_share, axis=1, raw=True)

    

    print('tfidf...')

    question1_tfidf = tfidf.transform(df_features.question1.tolist())

    question2_tfidf = tfidf.transform(df_features.question2.tolist())

    df_features['z_tfidf_sum1'] = np.sum(question1_tfidf, axis = 1)

    df_features['z_tfidf_sum2'] = np.sum(question2_tfidf, axis = 1)

    df_features['z_tfidf_mean1'] = np.mean(question1_tfidf, axis = 1)

    df_features['z_tfidf_mean2'] = np.mean(question2_tfidf, axis = 1)

    df_features['z_tfidf_len1'] = (question1_tfidf != 0).sum(axis = 1)

    df_features['z_tfidf_len2'] = (question2_tfidf != 0).sum(axis = 1)

    

    return df_features.fillna(0.0)
df_train = get_features(df_train)
df_train.head()
col = [c for c in df_train.columns if c[:1]=='z']
pos_train = df_train[df_train['is_duplicate'] == 1]

neg_train = df_train[df_train['is_duplicate'] == 0]

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

df_train = pd.concat([pos_train, neg_train])
x_train, x_valid, y_train, y_valid = train_test_split(df_train[col], df_train['is_duplicate'], test_size=0.2, random_state=0)
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

bst = xgb.train(params, d_train, 500, watchlist, 

                early_stopping_rounds=50, 

                verbose_eval=100) #change to higher #s
print(log_loss(df_train.is_duplicate, bst.predict(xgb.DMatrix(df_train[col]))))
df_test = get_features(df_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = bst.predict(xgb.DMatrix(df_test[col]))
sub.to_csv('zmix_submission_xgb_01.csv', index=False)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



plt.rcParams['figure.figsize'] = (7.0, 7.0)

xgb.plot_importance(bst); plt.show()
