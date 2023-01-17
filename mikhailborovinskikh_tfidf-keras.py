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
from random import shuffle

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector

from keras.utils import np_utils

from keras.preprocessing import text, sequence
df_train_txt = pd.read_csv('../input/training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

df_train_var = pd.read_csv('../input/training_variants')

df_test_txt = pd.read_csv('../input/test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])

df_test_var = pd.read_csv('../input/test_variants')

training_merge_df = df_train_var.merge(df_train_txt,left_on="ID",right_on="ID")

testing_merge_df = df_test_var.merge(df_test_txt,left_on="ID",right_on="ID")
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
%%time

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
train_y = training_merge_df['Class'].values

label_encoder = LabelEncoder()

label_encoder.fit(train_y)

encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))
one_hot_gene = pd.get_dummies( np.hstack((training_merge_df['Gene'].values,testing_merge_df['Gene'].values)))

one_hot_variation = pd.get_dummies( np.hstack((training_merge_df['Variation'].values,testing_merge_df['Variation'].values)))
from scipy.sparse import hstack
# define model

def baseline_model():

    model = Sequential()

    model.add(Dense(512, input_dim=transformer_bag_of_words.shape[1]+one_hot_gene.shape[1]+one_hot_variation.shape[1], init='normal', activation='relu'))

    model.add(Dropout(0.15))

    model.add(Dense(512, init='normal', activation='relu'))

    model.add(Dropout(0.15))

    model.add(Dense(512, init='normal', activation='relu'))

    model.add(Dropout(0.15))

    model.add(Dense(512, init='normal', activation='relu'))

    model.add(Dense(256, init='normal', activation='relu'))

    model.add(Dense(64, init='normal', activation='relu'))

    model.add(Dense(9, init='normal', activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy'])

    return model





estimator = KerasClassifier(build_fn=baseline_model, epochs=15, batch_size=64)

estimator.fit(hstack((one_hot_gene[:training_merge_df.shape[0]], one_hot_variation[:training_merge_df.shape[0]], transformer_bag_of_words)).todense(), encoded_y, validation_split=0.05)
%%time

results = estimator.predict_proba(hstack((one_hot_gene[training_merge_df.shape[0]:], one_hot_variation[training_merge_df.shape[0]:], X_test_transformer)).todense())
results_df = pd.read_csv("../input/submissionFile")

for i in range(1,10):

    results_df['class'+str(i)] = results.transpose()[i-1]

results_df.to_csv('output_tf_keras_version2',sep=',',header=True,index=None)

results_df.head()