import numpy as np 

import pandas as pd 

import re

# timing function

import time   

start = time.clock() #_________________ measure efficiency timing





train=pd.read_csv('../input/train.csv')[:20000].fillna("")



def cleanup(x):

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x





def edit_distance(s1, s2):

    m=len(s1)+1

    n=len(s2)+1



    tbl = {}

    for i in range(m): tbl[i,0]=i

    for j in range(n): tbl[0,j]=j

    for i in range(1, m):

        for j in range(1, n):

            cost = 0 if s1[i-1] == s2[j-1] else 1

            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)



    return tbl[i,j]



def leve3(string_1, string_2):

    len_1 = len(ngrams_split(string_1,3)) + 1

    len_2 = len(ngrams_split(string_2,3)) + 1

    d=[0]

    if len_1>3 and len_2>3:

        d = [0] * (len_1 * len_2)



        for i in range(len_1):

            d[i] = i

        for j in range(len_2):

            d[j * len_1] = j



        for j in range(1, len_2):

            for i in range(1, len_1):

                if string_1[i - 3] == string_2[j - 3]:

                    d[i + j * len_1] = d[i - 1 + (j - 1) * len_1]

                else:

                    d[i + j * len_1] = min(

                       d[i - 1 + j * len_1] + 1,        # deletion

                       d[i + (j - 1) * len_1] + 1,      # insertion

                       d[i - 1 + (j - 1) * len_1] + 1,  # substitution

                    )



    return d[-1]



questions = train['question1'].tolist() + train['question2'].tolist()

train=cleanup(train)

print(train.head())

end = time.clock()

print('open:',end-start)
import nltk #language functions

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,laplacian_kernel,sigmoid_kernel,polynomial_kernel,rbf_kernel

from sklearn.decomposition import TruncatedSVD

import scipy



def ngram(lst):

    woorden=nltk.word_tokenize(lst.lower())

    gra_ret=[]

    for woo in woorden:

        zip_lst=list(woo)

        grams=zip(zip_lst, zip_lst[1:], zip_lst[2:])

        trigram=[]

        for gr in grams:

            trigram.append(''.join(gr))    

        gra_ret+=trigram



    return ' '.join(gra_ret)



def intersecting(a, b):

    return ' '.join(list(set(a.split()) & set(b.split())))



def differencing(a, b):

    return ' '.join(list(set(a.split()) ^ set(b.split())))





def get_fea(df_fea):

    print('3gramming')

    df_fea['q13g'] = df_fea['question1'].apply(lambda x: ngram(x))

    df_fea['q23g'] = df_fea['question2'].apply(lambda x: ngram(x))        

    df_fea['inte'] = df_fea[['q13g','q23g']].apply(lambda x: intersecting(*x), axis=1)

    df_fea['diffe'] = df_fea[['q13g','q23g']].apply(lambda x: differencing(*x), axis=1)    

    df_fea['q1di'] = df_fea[['q13g','diffe']].apply(lambda x: intersecting(*x), axis=1)    

    df_fea['q2di'] = df_fea[['q23g','diffe']].apply(lambda x: intersecting(*x), axis=1)        

        

    return df_fea.fillna(0.0)

    

df_train = get_fea(train)

#print(df_train)

end = time.clock()

print('gramming:',len(df_train)*1.0/(end-start))



questions = train['q13g'].tolist() + train['q23g'].tolist()

tfidf = TfidfVectorizer( ngram_range=(2, 3))

tfidf.fit_transform(questions)



print(tfidf)



def get_feaT(df_feaT):

    question1_tfidf = tfidf.transform(df_feaT.q13g.tolist())

    print('Q1')    

    question2_tfidf = tfidf.transform(df_feaT.q23g.tolist())    

    print('Q2')        

    questionI_tfidf = tfidf.transform(df_feaT.inte.tolist())    

    questionD_tfidf = tfidf.transform(df_feaT.diffe.tolist()) 

    questionQ1D_tfidf = tfidf.transform(df_feaT.q1di.tolist())    

    questionQ2D_tfidf = tfidf.transform(df_feaT.q2di.tolist())  



    print('sum mean len....')

    df_feaT['tfidfSum1'] = scipy.sparse.csr_matrix(question1_tfidf).sum(axis=1)

    df_feaT['tfidfSum2'] = scipy.sparse.csr_matrix(question2_tfidf).sum(axis=1)

    df_feaT['tfidfSumI'] = scipy.sparse.csr_matrix(questionI_tfidf).sum(axis=1)   

    df_feaT['tfidfSumD'] = scipy.sparse.csr_matrix(questionD_tfidf).sum(axis=1)

    df_feaT['tfidfSum1D'] = scipy.sparse.csr_matrix(questionQ1D_tfidf).sum(axis=1)     

    df_feaT['tfidfSum2D'] = scipy.sparse.csr_matrix(questionQ2D_tfidf).sum(axis=1)    

    

    df_feaT['tfidfMean1'] = scipy.sparse.csr_matrix(question1_tfidf).mean(axis=1)

    df_feaT['tfidfMean2'] = scipy.sparse.csr_matrix(question2_tfidf).mean(axis=1)

    df_feaT['tfidfMeanI'] = scipy.sparse.csr_matrix(questionI_tfidf).mean(axis=1)    

    df_feaT['tfidfMeanD'] = scipy.sparse.csr_matrix(questionD_tfidf).mean(axis=1)    

    df_feaT['tfidfMean1D'] = scipy.sparse.csr_matrix(questionQ1D_tfidf).mean(axis=1)    

    df_feaT['tfidfMean2D'] = scipy.sparse.csr_matrix(questionQ2D_tfidf).mean(axis=1)    

    

    df_feaT['tfidfLen1'] = (question1_tfidf != 0).sum(axis = 1)

    df_feaT['tfidfLen2'] = (question2_tfidf != 0).sum(axis = 1)

    df_feaT['tfidfLenI'] = (questionI_tfidf != 0).sum(axis = 1)

    df_feaT['tfidfLenD'] = (questionD_tfidf != 0).sum(axis = 1)

    df_feaT['tfidfLen1D'] = (questionQ1D_tfidf != 0).sum(axis = 1)

    df_feaT['tfidfLen2D'] = (questionQ2D_tfidf != 0).sum(axis = 1)

    

    #(question1_tfidf.getrow())*(question1_tfidf.getrow())

    print('simil')    

    df_feaT['sim12'] = df_feaT['id'].apply(lambda i: (question1_tfidf.getrow(i)*question2_tfidf.getrow(i).T).toarray()[0][0])    

    df_feaT['sim1I'] = df_feaT['id'].apply(lambda i: (question1_tfidf.getrow(i)*questionI_tfidf.getrow(i).T).toarray()[0][0] )      

    df_feaT['sim1D'] = df_feaT['id'].apply(lambda i: (question1_tfidf.getrow(i)*questionD_tfidf.getrow(i).T).toarray()[0][0]  )      

    df_feaT['sim2I'] = df_feaT['id'].apply(lambda i: (question2_tfidf.getrow(i)*questionI_tfidf.getrow(i).T).toarray()[0][0]  )      

    df_feaT['sim2D'] = df_feaT['id'].apply(lambda i: (question2_tfidf.getrow(i)*questionD_tfidf.getrow(i).T).toarray()[0][0]  )          

    df_feaT['sim11D'] = df_feaT['id'].apply(lambda i: (question1_tfidf.getrow(i)*questionQ1D_tfidf.getrow(i).T).toarray()[0][0] )       

    df_feaT['sim22D'] = df_feaT['id'].apply(lambda i: (question2_tfidf.getrow(i)*questionQ2D_tfidf.getrow(i).T).toarray()[0][0] )           

    

    #df_feaT['cos12'] = df_feaT['id'].apply(lambda i: cosine_similarity(question1_tfidf.getrow(i),question2_tfidf.getrow(i))[0][0]) 

    #df_feaT['cos1I'] = df_feaT['id'].apply(lambda i: cosine_similarity(question1_tfidf.getrow(i),questionI_tfidf.getrow(i))[0][0])      

    #df_feaT['cos1D'] = df_feaT['id'].apply(lambda i: cosine_similarity(question1_tfidf.getrow(i),questionD_tfidf.getrow(i))[0][0])         

    print('similD')      

    #df_feaT['cos2I'] = df_feaT['id'].apply(lambda i: cosine_similarity(question2_tfidf.getrow(i),questionI_tfidf.getrow(i))[0][0])      

    #df_feaT['cos2D'] = df_feaT['id'].apply(lambda i: cosine_similarity(question2_tfidf.getrow(i),questionD_tfidf.getrow(i))[0][0])          

    #df_feaT['cos11D'] = df_feaT['id'].apply(lambda i: cosine_similarity(question1_tfidf.getrow(i),questionQ1D_tfidf.getrow(i))[0][0])      

    #df_feaT['cos22D'] = df_feaT['id'].apply(lambda i: cosine_similarity(question2_tfidf.getrow(i),questionQ2D_tfidf.getrow(i))[0][0])      

    print('eucl') 

    df_feaT['euc12'] = df_feaT['id'].apply(lambda i: euclidean_distances(question1_tfidf.getrow(i),question2_tfidf.getrow(i))[0][0]) 

    df_feaT['euc1D'] = df_feaT['id'].apply(lambda i: euclidean_distances(question1_tfidf.getrow(i),questionD_tfidf.getrow(i))[0][0])         

    df_feaT['euc2D'] = df_feaT['id'].apply(lambda i: euclidean_distances(question2_tfidf.getrow(i),questionD_tfidf.getrow(i))[0][0])         

    print('svd')     

    svd = TruncatedSVD(n_components=20, n_iter=30, random_state=42)

    tempi=pd.DataFrame(svd.fit_transform(questionI_tfidf))

    tempi.rename(columns=lambda x: str(x)+'_i', inplace=True) #nog eens zoeken omcolumns te renamen

    df_feaT=df_feaT.join(tempi,how='inner')

    print('tempi',tempi.shape)

    

    svd = TruncatedSVD(n_components=20, n_iter=30, random_state=42)

    tempd=pd.DataFrame(svd.fit_transform(questionD_tfidf))

    tempd.rename(columns=lambda x: str(x)+'_d', inplace=True) #nog eens zoeken omcolumns te renamen

    df_feaT=df_feaT.join(tempd,how='inner')

    print('tempd',tempd.shape)

    

    return df_feaT



df_train = get_feaT(df_train)

end = time.clock()

print('tfidf-sim:',len(df_train)*1.0/(end-start))



print(df_train.head(10))



y=train['is_duplicate']        

feats = df_train.columns.values.tolist()

feats=[x for x in feats if x not in ['question1','question2','q13g','q23g','inte','diffe','q1di','q2di','id','qid1','qid2','is_duplicate']]

print("features",feats)

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

import scipy

import xgboost as xgb

import difflib



x_train, x_valid, y_train, y_valid = train_test_split(df_train[feats], y, test_size=0.1, random_state=0)

#XGBoost model

params = {"objective":"binary:logistic",'eval_metric':'logloss',"max_depth":7}



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



sub = pd.DataFrame() # Submission data frame

sub['test_id'] = []

sub['is_duplicate'] = []

header=['test_id','question1','question2','id','qid1','qid2','is_duplicate']

test=pd.read_csv('../input/test.csv')[:10000].fillna("")

test.columns=['id','question1','question2']



print("cleaning test")

df_test=cleanup(test)

print('cleaned',df_test.head())

df_test = get_fea(df_test)

df_test = get_feaT(df_test)

print('engineered',df_test.head())



sub=pd.DataFrame({'test_id':df_test['id'], 'is_duplicate':bst.predict(xgb.DMatrix(df_test[feats]))})

print(sub.head())



sub.to_csv('../quora_submission_svd_xgb.csv', index=False)