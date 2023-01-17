# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import pandas as pd

import numpy as np



import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.decomposition import TruncatedSVD

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfTransformer

import nltk

import re

from nltk.corpus import stopwords

import os
df_train_txt = pd.read_csv('../input/training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

df_train_var = pd.read_csv('../input/training_variants')

df_test_txt = pd.read_csv('../input/test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

df_test_var = pd.read_csv('../input/test_variants')

training_merge_df = df_train_var.merge(df_train_txt,left_on="ID",right_on="ID")

testing_merge_df = df_test_var.merge(df_test_txt,left_on="ID",right_on="ID")
training_merge_df.head()
def textClean(text):

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = text.lower().split()

    stops = {'so', 'his', 't', 'y', 'ours', 'herself', 

             'your', 'all', 'some', 'they', 'i', 'of', 'didn', 

             'them', 'when', 'will', 'that', 'its', 'because', 

             'while', 'those', 'my', 'don', 'again', 'her', 'if',

             'further', 'now', 'does', 'against', 'won', 'same', 

             'a', 'during', 'who', 'here', 'have', 'in', 'being', 

             'it', 'other', 'once', 'itself', 'hers', 'after', 're',

             'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 

             'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',

             'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',

             'the', 'until', 'she', 'am', 'under', 'how', 'yourself',

             'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',

             'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 

             'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',

             'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 

             'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',

             'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 

             'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 

             'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',

             'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 

             'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',

             'o', 'before'}

    text = [w for w in text if not w in stops]    

    text = " ".join(text)

    text = text.replace("."," ").replace(","," ")

    return(text)
trainText = []

for it in training_merge_df['Text']:

    newT = textClean(it)

    trainText.append(newT)

testText = []

for it in testing_merge_df['Text']:

    newT = textClean(it)

    testText.append(newT)
from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()

for i in range(len(trainText)):

    trainText[i] = st.stem(trainText[i])

for i in range(len(testText)):

    testText[i] = st.stem(testText[i])
%%time

#I used CuntVectorizer before, best result is 0.77.

#Now I use TfIdfVectorizer, best result iz 0.67 with ngram (1,2).

#I think that ngram (1,3) may be better *)

#count_vectorizer = CountVectorizer(min_df=5, ngram_range=(1,2), max_df=0.65,

                       #tokenizer=nltk.word_tokenize,

                       #strip_accents='unicode',

                       #lowercase =True, analyzer='word', token_pattern=r'\w+',

                       #stop_words = 'english')

count_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.65,

                        tokenizer=nltk.word_tokenize,

                        strip_accents='unicode',

                        lowercase =True, analyzer='word', token_pattern=r'\w+',

                        use_idf=True, smooth_idf=True, sublinear_tf=False, 

                        stop_words = 'english')

bag_of_words = count_vectorizer.fit_transform(trainText)

print(bag_of_words.shape)

X_test = count_vectorizer.transform(testText)

print(X_test.shape)
%%time

transformer = TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=False)

transformer_bag_of_words = transformer.fit_transform(bag_of_words)

X_test_transformer = transformer.transform(X_test)

print (transformer_bag_of_words.shape)

print (X_test_transformer.shape)
gene_le = LabelEncoder()

gene_encoded = gene_le.fit_transform( np.hstack((training_merge_df['Gene'].values.ravel(),testing_merge_df['Gene'].values.ravel()))).reshape(-1, 1)

gene_encoded = gene_encoded / float(np.max(gene_encoded))





variation_le = LabelEncoder()

variation_encoded = variation_le.fit_transform( np.hstack((training_merge_df['Variation'].values.ravel(),testing_merge_df['Variation'].values.ravel()))).reshape(-1, 1)

variation_encoded = variation_encoded / float(np.max(variation_encoded))
from scipy.sparse import hstack
%%time

#This for (1,1) ngram lambda l2 and num leaves 50 (95), 

#num iterations 1000 (500), learning rate 0.01(0.05), max_depth 7 (5)

params = {'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'multiclass',

    'num_class': 9,

    'metric': {'multi_logloss'},

    'learning_rate': 0.01, 

    'max_depth': 10,

    'num_iterations': 1500, 

    'num_leaves': 55, 

    'min_data_in_leaf': 66, 

    'lambda_l2': 1.0,

    'feature_fraction': 0.8, 

    'bagging_fraction': 0.8, 

    'bagging_freq': 5}



x1, x2, y1, y2 = train_test_split(hstack((gene_encoded[:training_merge_df.shape[0]], variation_encoded[:training_merge_df.shape[0]], transformer_bag_of_words)), training_merge_df['Class'].values.ravel()-1, test_size=0.1, random_state=1)

d_train = lgb.Dataset(x1, label=y1)

d_val = lgb.Dataset(x2, label=y2)



model = lgb.train(params, train_set=d_train, num_boost_round=280,

               valid_sets=[d_val], valid_names=['dval'], verbose_eval=20,

               early_stopping_rounds=20)
%%time

results = model.predict(hstack((gene_encoded[training_merge_df.shape[0]:], variation_encoded[training_merge_df.shape[0]:], X_test_transformer)))
results_df = pd.read_csv("../input/submissionFile")

for i in range(1,10):

    results_df['class'+str(i)] = results.transpose()[i-1]

results_df.to_csv('output_tf_one_hot11',sep=',',header=True,index=None)

results_df.head()