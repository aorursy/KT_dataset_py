# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
import random
import os

pd.set_option('max_columns',None)
pd.options.display.width = 2000
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)
print(os.listdir("../input"))

df_kp = pd.read_csv("../input/Youtube02-KatyPerry.csv")
df_em = pd.read_csv("../input/Youtube04-Eminem.csv")
df_psy = pd.read_csv("../input/Youtube01-Psy.csv")
df_sh = pd.read_csv("../input/Youtube05-Shakira.csv")
df_lm = pd.read_csv("../input/Youtube03-LMFAO.csv")

df_main = pd.concat([df_kp,df_em,df_psy,df_sh,df_lm], ignore_index=True, verify_integrity=True)
print("Total size of dataset",len(df_main))
#print("Null values",df_main.isnull().sum())
display(df_main.head())
df_tr,df_te = train_test_split(df_main,train_size=0.7,test_size=0.3)
print("Train & Test size",df_tr.shape[0],df_te.shape[0])

count_vec = CountVectorizer(stop_words="english",lowercase=True) #,ngram_range=(1,3) reduced accuracy
count_mat = count_vec.fit_transform(df_tr["CONTENT"])
tf_idf_transform = TfidfTransformer(use_idf=True) #,sublinear_tf=True
tr_tf_idf = tf_idf_transform.fit_transform(count_mat) #(1369, 3214)

count_mat_te = count_vec.transform(df_te["CONTENT"])
tf_idf_te = tf_idf_transform.transform(count_mat_te)
print("vector shape",tr_tf_idf.shape)
#Naive-Bayes
clf = MultinomialNB().fit(tr_tf_idf, df_tr["CLASS"])
y = clf.predict(tf_idf_te)
print("Bayesian Accuracy",np.mean(np.squeeze(np.asarray(df_te["CLASS"].values)) == y)*100)
#df_res = pd.DataFrame({"Comment":df_te["CONTENT"],"PredClass":y,"RealClass":df_te["CLASS"]})
#display(df_res[df_res["PredClass"] != df_res["RealClass"]])

#Linear SVC
svc_clf = LinearSVC()
svc_clf.fit(tr_tf_idf,df_tr["CLASS"])
y = svc_clf.predict(tf_idf_te)
print("SVC Accuracy",np.mean(np.squeeze(np.asarray(df_te["CLASS"].values)) == y)*100)

#Try NLTK,stemming,gensim word2Vec

#Neural Net
clf = Sequential()
clf.add(Dense(4000,input_shape=(tr_tf_idf.shape[1],)))
clf.add(Dense(1000))
clf.add(Dense(1, activation='sigmoid'))
clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.fit(tr_tf_idf,df_tr["CLASS"],epochs=9,verbose=1)
scores = clf.evaluate(tf_idf_te,df_te["CLASS"])
print("NN Accuracy",scores[1]*100)
# Any results you write to the current directory are saved as output.
