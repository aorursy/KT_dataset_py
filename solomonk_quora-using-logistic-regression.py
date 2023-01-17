%reset -f

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from nltk import word_tokenize

from  datetime import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

import time

#from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from nltk.stem.porter import *

stemmer = PorterStemmer()

import random

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier



import scipy.io

import scipy.io

import scipy.io

# import pysptk

import scipy.io.wavfile

import warnings



warnings.filterwarnings('ignore')



# numerical processing and scientific libraries

import scipy



from sklearn.cross_validation import StratifiedKFold

from scipy.stats import norm

import scipy.sparse

from sklearn.feature_extraction.text import TfidfVectorizer



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams

import codecs, difflib, Levenshtein

import random

import numpy as np

import sys

import random

from  datetime import *

import numpy as np

import os

from sklearn.metrics import roc_curve

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import pandas

from sklearn.metrics import log_loss

from sklearn.preprocessing import PolynomialFeatures

import time

import datetime

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.cross_validation import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

eng_stopwords = set(stopwords.words('english'))



df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")



num_train = df_train.shape[0]

print (num_train)
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



df_all_orig=df_all.copy(deep=True)
df_all.head()


df_all = df_all.drop(['id','qid1','qid2','question1','question2','questions'],axis=1)



df_train = df_all.iloc[:num_train]

df_test = df_all.iloc[num_train:]

id_test = df_test['test_id']



y_train = df_train['is_duplicate'].values

X_train = df_train.drop(['test_id','is_duplicate'],axis=1).values

X_test = df_test.drop(['test_id','is_duplicate'],axis=1).values



from sklearn.cross_validation import train_test_split

import xgboost as xgb



X_df_train_SINGLE = X_train

answers_1_SINGLE = list(y_train)



#X_df_train_SINGLE = X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

trainX, testX, trainY, testY = train_test_split(X_df_train_SINGLE, answers_1_SINGLE, test_size=.22)  # CV

        

    

xgbm = xgb.XGBClassifier(base_score=0.5, colsample_bytree=0.5,

                                       gamma=0.017, learning_rate=0.15, max_delta_step=0,

                                       max_depth=100, min_child_weight=3, n_estimators=500,

                                       nthread=-1, objective='binary:logistic', seed=0,

                                       silent=1, subsample=0.9)



print ('Running:' + str(xgbm) + 'shape:' + str(X_df_train_SINGLE.shape))    



model_train = xgbm.fit(trainX, trainY, early_stopping_rounds=100, 

                       eval_metric="logloss",eval_set=[(testX, testY)], 

                       verbose=True)



# print model_train

predictions = xgbm.predict_proba(testX)[:, 1]



print ('ROC AUC:' + str(roc_auc_score(testY, predictions)))

print ('LOG LOSS:' + str(log_loss(testY, predictions)))
d_test = xgb.DMatrix(X_test)

p_test = xgbm.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = np.int32(id_test)

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)