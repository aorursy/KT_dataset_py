import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

import string

from gensim.models import Word2Vec

from gensim import corpora

from gensim import models



from gensim.matutils import jaccard, cossim

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

import xgboost as xgb

from sklearn.cross_validation import train_test_split

import os.path





import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)





stop = stopwords.words('english')



def preprocess(data):

    return data.lower().split()



train = pd.read_csv("../input/train.csv", keep_default_na='')

test = pd.read_csv("../input/test.csv", keep_default_na='')



combine = [train, test]



print("Preprocessing questions...")    

for dataset in combine:

    dataset['processed_q1'] = dataset['question1'].apply(preprocess)

    dataset['processed_q2'] = dataset['question2'].apply(preprocess)





all_questions = train['processed_q1'] + train['processed_q1'] + test['processed_q1'] + test['processed_q2']

all_questions = all_questions[all_questions.notnull()]



dict_file = 'quora.dict'

corpus_file = 'corpus.mm'

word_vec_file = 'word2vec.model'





if os.path.isfile(dict_file):

    print("Loading dictionary file")

    dictionary = corpora.Dictionary.load(dict_file)

else:    

    print("Generating dictionary file")

    dictionary = corpora.Dictionary(all_questions)

    dictionary.save(dict_file)



if os.path.isfile(corpus_file):

    print("Loading corpus file")

    corpus = corpora.MmCorpus(corpus_file)

else:    

    print("Generating corpus file")

    corpus = [dictionary.doc2bow(question) for question in all_questions]

    corpora.MmCorpus.serialize(corpus_file, corpus) 



if os.path.isfile(word_vec_file):

    print("Loading word 2 vect file")

    w2v_model = Word2Vec.load(word_vec_file)

else:

    print("Generating word 2 vect file")

    w2v_model = Word2Vec(all_questions, min_count=10)

    w2v_model.save(word_vec_file)

    

def to_bow(doc):

    return dictionary.doc2bow(doc)



def filter_in_vacob(doc):

    return list(filter(lambda x: x in w2v_model, doc))



print("Computing similarity metrics")

for dataset in combine:

    dataset['jaccard'] = dataset.apply(lambda d: jaccard(to_bow(d['processed_q1']), to_bow(d['processed_q2'])), axis=1)

    dataset['cosine'] = dataset.apply(lambda d: cossim(to_bow(d['processed_q1']), to_bow(d['processed_q2'])), axis=1)

    dataset['wv_doc_1'] = dataset['processed_q1'].apply(lambda d:filter_in_vacob(d))

    dataset['wv_doc_2'] = dataset['processed_q2'].apply(lambda d:filter_in_vacob(d))

    dataset['wv_sim'] = dataset.apply(lambda d: w2v_model.wv.n_similarity(d['wv_doc_1'], d['wv_doc_2']) if d['wv_doc_1'] and d['wv_doc_2'] else -1, axis=1)



    

x_train = pd.DataFrame()

x_test = pd.DataFrame()



x_train['jaccard'] = train['jaccard']

x_train['cosine'] = train['cosine']

x_train['wv_sim'] = train['wv_sim']



x_test['jaccard'] = test['jaccard']

x_test['cosine'] = test['cosine']

x_test['wv_sim'] = test['wv_sim']



y_train = train['is_duplicate']





x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    
params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict_proba(x_test)

print(random_forest.score(x_train, y_train))



y_pred_df = pd.DataFrame({'0':y_pred[:,0],'1':y_pred[:,1]})

sub = pd.DataFrame()

sub['test_id'] = test['test_id']

sub['is_duplicate'] = y_pred_df['1']

sub.to_csv('random_forest_probability.csv', index=False)