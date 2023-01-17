# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import fetch_20newsgroups
train=fetch_20newsgroups(data_home='.',subset='train')

test=fetch_20newsgroups(data_home='.',subset='test')
train.keys(),test.keys()
len(train['data']),len(test['data'])
for i ,label in enumerate (train['target_names']):

    print(f'class{i:2d}={label}')
item_num=25

class_num=train['target'][item_num]

print(f'Class number={class_num}')

print(f'Class name={train["target_names"][class_num]}')

print(train['data'][item_num])
item_num=25

class_num=test['target'][item_num]

print(f'Class number={class_num}')

print(f'Class name={test["target_names"][class_num]}')

print(test['data'][item_num])
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB



#Create the DTM first

cv=CountVectorizer(stop_words='english')

train_dtm=cv.fit_transform(train['data'])

test_dtm=cv.transform(test['data'])



#Fit the model

nb=MultinomialNB()

nb=nb.fit(train_dtm,train['target'])

#predict

predicted=nb.predict(test_dtm)

score=nb.score(test_dtm,test['target'])

print('Accuracy of Naive Bayes :')

print(score*100.0)
#Classification Report

from sklearn import metrics

print(metrics.classification_report(test['target'],predicted,target_names=test['target_names']))
#Confusion Matrix

from helper_code import mlplots as ml

fig,ax=plt.subplots(figsize=(13,10))

ml.confusion(test['target'],predicted,test['target_names'],'Naive Bayes Model')
#or

from mlplot.evaluation import ClassificationEvaluation

eval = ClassificationEvaluation(test['target'], predicted,test['target_names'],'Naive Bayes')

eval.confusion_matrix(threshold=0.5)



#confusion matrix
from sklearn.feature_extraction.text import TfidfVectorizer



tf_cv=TfidfVectorizer(stop_words='english')

train_dtm_tf=tf_cv.fit_transform(train['data'])

test_dtm_tf=tf_cv.transform(test['data'])



nb.fit(train_dtm_tf,train['target'])

prdicted=nb.predict(test_dtm_tf)

print("Accuracy of Naive Bayes Algo:")

score=100.0* nb.score(test_dtm_tf,test['target'])

print(score)