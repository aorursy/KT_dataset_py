# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle 

import nltk

import time

from nltk.corpus import stopwords

from sklearn.datasets import load_files

import seaborn as sns

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import re

import nltk as nlp

nltk.download("stopwords")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Lets import the csv file in pandas dataframe first

data = pd.read_csv('/kaggle/input/ttc4900/7allV03.csv', encoding='utf-8', header=None, names=['cat', 'text'])

data
print(len(data))
data.info()
print(data.cat.unique())
data['labels'] = pd.factorize(data.cat)[0]
data.head()
data.drop([0],axis=0,inplace=True)
data.head()
teknoloji = data[data.labels == 7]

spor = data[data.labels == 6]

saglik = data[data.labels == 5]

kultur = data[data.labels == 4]

ekonomi = data[data.labels == 3]

dunya = data[data.labels == 2]

siyaset = data[data.labels == 1]
spor.info()
teknoloji.info()
sns.countplot(x="labels", data=data)

data.loc[:,'labels'].value_counts()
text_list = []

for text in data.text: 

    text = text.lower()  #Büyük harften -Küçük harfe çevirme

    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]"," ",text) # a-z dışındakileri boşlukla değiştir

    text =nltk.word_tokenize(text) 

    text =[word for word in text if not word in set(stopwords.words("turkish"))] #gereksiz kelimeleri çıkarma 

    lemma = nlp.WordNetLemmatizer() #lemmatazation

    text = [lemma.lemmatize(word) for word in text] # Köklerini bulma

    text = " ".join(text) # boşlukla birleştir tüm kelimeleri

    text_list.append(text) # text_list'i doldur

text_list
from sklearn.feature_extraction.text import CountVectorizer

max_features = 1000

count_vectorizer = CountVectorizer(max_features=max_features)
sparce_matrix = count_vectorizer.fit_transform(text_list).toarray() #x 

text_list
print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))
# creating x and y 

y = data.iloc[:,2].values

x = sparce_matrix
#train-test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state = 40)
from sklearn.naive_bayes import MultinomialNB

Mnb = MultinomialNB()

Mnb.fit(x_train,y_train)
y_pred = Mnb.predict(x_test)

print("Naive Bayes algorithm accuracy: ",Mnb.score(x_test,y_test))
x.shape
y_pred
#CONFUSİON MATRİX FOR NAIVE BAYES

y_pred = Mnb.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 40)

#%%

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("Decision Tree algorithm accuracy: ", dt.score(x_test,y_test))

#CONFUSİON MATRİX FOR DESICION TREE

y_pred = dt.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 50,random_state = 1)

rf.fit(x_train,y_train)

print("Random Forest algorithm accuracy: ",rf.score(x_test,y_test))
#CONFUSİON MATRİX FOR RANDOM FOREST

y_pred = rf.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=50)

training_start = time.perf_counter()

xgb.fit(x_train, y_train)

preds = xgb.predict(x_test)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

prediction_end = time.perf_counter()

acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100

xgb_train_time = training_end-training_start

xgb_prediction_time = prediction_end-prediction_start

print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))

print("Time consumed for training: %4.3f" % (xgb_train_time))

print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))
#CONFUSİON MATRİX FOR XGBClassifier

#preds = xgb.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)
f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()