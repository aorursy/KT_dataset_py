import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

#from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

from nltk.stem.porter import *

stemmer = PorterStemmer()

import random

import re



random.seed(1337)



df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")



num_train = df_train.shape[0]

print (num_train)
df_train.head()
df_test.head()
def str_stem(str1):

    str1 = str(str1)

    str1 = re.sub(r'[^a-zA-Z0-9 ]',r'',str1)

    str1 = str1.lower()

    #str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])

    return str1



def str_common_word(str1, str2):

    str1, str2 = str1.lower(), str2.lower()

    words, cnt = str1.split(), 0

    for word in words:

        if str2.find(word)>=0:

            cnt+=1

    return cnt

def ngram(tokens, n):

    grams =[tokens[i:i+n] for i in range(len(tokens)-(n-1))]

    return grams



def get_sim(a_tri,b_tri):

    intersect = len(set(a_tri) & set(b_tri))

    union = len(set(a_tri) | set(b_tri))

    if union == 0:

        return 0

    return float(intersect)/(union)



def jaccard_similarity(str1,str2):

    sentence_gram1 = str1

    sentence_gram2 = str2

    grams1 = ngram(sentence_gram1, 5)

    grams2 = ngram(sentence_gram2, 5)

    similarity = get_sim(grams1, grams2)

    return similarity

    

    

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)





df_all['question1'] = df_all['question1'].map(lambda x:str_stem(x))

df_all['question2'] = df_all['question2'].map(lambda x:str_stem(x))



df_all['len_of_q1'] = df_all['question1'].map(lambda x:len(x.split())).astype(np.int64)

df_all['len_of_q2'] = df_all['question2'].map(lambda x:len(x.split())).astype(np.int64)



df_all['questions'] = df_all['question1']+"|"+df_all['question2']

print ("Questions combined...")

df_all['q2_in_q1'] = df_all['questions'].map(lambda x:str_common_word(x.split('|')[0],x.split('|')[1]))

df_all['q1_in_q2'] = df_all['questions'].map(lambda x:str_common_word(x.split('|')[1],x.split('|')[0]))

print ("Common words found ...")

df_all['jaccard'] = df_all['questions'].map(lambda x:jaccard_similarity(x.split('|')[0],x.split('|')[1]))

print ("Jaccard similarities computed...")

#df_all['lev_distance'] = df_all['questions'].map(lambda x:normalized_damerau_levenshtein_distance(x.split('|')[0],x.split('|')[1]))

#print ("Levenshtein distances computed...")
df_all.head()


df_all = df_all.drop(['id','qid1','qid2','question1','question2','questions'],axis=1)



df_train = df_all.iloc[:num_train]

df_test = df_all.iloc[num_train:]

id_test = df_test['test_id']



y_train = df_train['is_duplicate'].values

X_train = df_train.drop(['test_id','is_duplicate'],axis=1).values

X_test = df_test.drop(['test_id','is_duplicate'],axis=1).values



from sklearn.cross_validation import train_test_split



x_trainb, x_validb, y_trainb, y_validb = train_test_split(X_train, y_train, test_size=0.2, random_state=4747)



import xgboost as xgb



params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_trainb, label=y_trainb)

d_valid = xgb.DMatrix(x_validb, label=y_validb)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(X_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = np.int32(id_test)

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)