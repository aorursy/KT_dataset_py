import numpy as np 

import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

import scipy

import xgboost as xgb

import difflib

import re

from nltk.corpus import stopwords

from nltk.metrics import jaccard_distance





#Reading and processing of data

train=pd.read_csv('../input/train.csv')[:20000].fillna("")

#train=pd.read_csv('../input/train.csv').dropna()

stops = set(stopwords.words("english"))

y=train['is_duplicate']

train=train.drop(['id', 'qid1', 'is_duplicate','qid2'], axis=1)



#Cleaning up the data

#Removing ? mark and non ASCII characters

def cleanup(data):

    data['question1'] = data['question1'].apply(lambda x: x.rstrip('?'))

    data['question2'] = data['question2'].apply(lambda x: x.rstrip('?'))

    # Removing non ASCII chars

    data['question1']=data['question1'].apply(lambda x: x.replace(r'[^\x00-\x7f]',r' '))

    data['question2']=data['question2'].apply(lambda x: x.replace(r'[^\x00-\x7f]',r' ')) 

    # Pad punctuation with spaces on both sides

    '''

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    '''

    return data



questions = train['question1'].tolist() + train['question2'].tolist()

train=cleanup(train)

print(train.head())
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,laplacian_kernel,sigmoid_kernel,polynomial_kernel,rbf_kernel

from sklearn.decomposition import TruncatedSVD



def intersecting(a, b):

    return ' '.join(list(set(a.split()) & set(b.split())))



def differencing(a, b):

    return ' '.join(list(set(a.split()) ^ set(b.split())))



tfidf = TfidfVectorizer( ngram_range=(1, 3))

tfidf.fit_transform(questions)

#print(tfidf.vocabulary_)



def get_features(df_features):    

    df_features['interseq'] = df_features[['question1','question2']].apply(lambda x: intersecting(*x), axis=1)

    df_features['diffseq'] = df_features[['question1','question2']].apply(lambda x: differencing(*x), axis=1)    

    df_features['q1d'] = df_features[['question1','diffseq']].apply(lambda x: intersecting(*x), axis=1)    

    df_features['q2d'] = df_features[['question2','diffseq']].apply(lambda x: intersecting(*x), axis=1)        

    

    print('tfidf')    

    question1_tfidf = tfidf.transform(df_features.question1.tolist())

    question2_tfidf = tfidf.transform(df_features.question2.tolist())    

    questionI_tfidf = tfidf.transform(df_features.interseq.tolist())    

    questionD_tfidf = tfidf.transform(df_features.diffseq.tolist()) 

    questionQ1D_tfidf = tfidf.transform(df_features.q1d.tolist())    

    questionQ2D_tfidf = tfidf.transform(df_features.q2d.tolist()) 

    print(question1_tfidf.shape)

        

    print('SVD')

    svd = TruncatedSVD(n_components=30)

    df_features=df_features.join(pd.DataFrame(svd.fit_transform(questionI_tfidf)),how='inner')

    print(svd.explained_variance_ratio_)

    print(svd.get_params(deep=True))

    

    svd = TruncatedSVD(n_components=30)

    temp=pd.DataFrame(svd.fit_transform(questionD_tfidf))

    temp.rename(columns=lambda x: str(x)+'_d', inplace=True) #nog eens zoeken omcolumns te renamen

    df_features=df_features.join(temp,how='inner')  



    

    svd = TruncatedSVD(n_components=30)

    temp=pd.DataFrame(svd.fit_transform(question1_tfidf))

    temp.rename(columns=lambda x: str(x)+'_q1', inplace=True) #nog eens zoeken omcolumns te renamen

    df_features=df_features.join(temp,how='inner') 



    print(temp.shape)

    

    svd = TruncatedSVD(n_components=30)

    temp=pd.DataFrame(svd.fit_transform(question2_tfidf))

    temp.rename(columns=lambda x: str(x)+'_q2', inplace=True) #nog eens zoeken omcolumns te renamen

    df_features=df_features.join(temp,how='inner')     

 

    svd = TruncatedSVD(n_components=30)

    temp=pd.DataFrame(svd.fit_transform(questionQ1D_tfidf))

    temp.rename(columns=lambda x: str(x)+'_q1d', inplace=True) #nog eens zoeken omcolumns te renamen

    df_features=df_features.join(temp,how='inner')   

    

    svd = TruncatedSVD(n_components=30)

    temp=pd.DataFrame(svd.fit_transform(questionQ2D_tfidf))

    temp.rename(columns=lambda x: str(x)+'_q2d', inplace=True) #nog eens zoeken omcolumns te renamen

    df_features=df_features.join(temp,how='inner')   

    

    return df_features.fillna(0.0)







df_train = get_features(train)

feats = df_train.columns.values.tolist()

feats=[x for x in feats if x not in ['question1','question2','Q1seq','Q2seq','q1d','q2d','interseq','diffseq','id','qid1','qid2','is_duplicate']]

print("features",feats)

print(df_train.head())
x_train, x_valid, y_train, y_valid = train_test_split(df_train[feats], y, test_size=0.3, random_state=0)

#XGBoost model

params = {"objective":"binary:logistic",'eval_metric':'logloss',"eta": 0.11,

          "subsample":0.7,"min_child_weight":1,"colsample_bytree": 0.7,

          "max_depth":5,"silent":1,"seed":2017}



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=200,verbose_eval=25) #change to higher #s

print('training done')



print("log loss for training data set",log_loss(y, bst.predict(xgb.DMatrix(df_train[feats]))))

#Predicting for test data set

sub = pd.DataFrame() # Submission data frame

sub['test_id'] = []

sub['is_duplicate'] = []

header=['test_id','question1','question2','id','qid1','qid2','is_duplicate']

test=pd.read_csv('../input/test.csv')[:20000].fillna("")

print("cleaning test")

df_test=cleanup(test)

print("feature engineering for test")

df_test = get_features(df_test)

sub=pd.DataFrame({'test_id':df_test['test_id'], 'is_duplicate':bst.predict(xgb.DMatrix(df_test[feats]))})

sub.to_csv('quora_submission_xgb_11.csv', index=False)