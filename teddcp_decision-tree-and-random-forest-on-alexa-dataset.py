import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
import os

os.listdir('../input/amazon-alexa-reviews')
data=pd.read_csv('../input/amazon-alexa-reviews/amazon_alexa.tsv',sep='\t')

data.head()
data.drop(columns=['date'],inplace=True)

data.head()
data.info()
data.describe()
sns.countplot(x='rating',data=data,hue='feedback')
sns.distplot(data['rating'])
sns.countplot(x='feedback',data=data)
plt.figure(figsize=(24,12))

sns.countplot(x='variation',hue='feedback',data=data)
from sklearn.feature_extraction.text import CountVectorizer as cv

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn.compose import ColumnTransformer as ct

from sklearn.pipeline import make_pipeline as mp

from sklearn.tree import DecisionTreeClassifier as dtc
## Important: i have passed the columns a string to CV and list of columns to OHE

transformer=ct(transformers=[('review_counts',cv(),'verified_reviews'), 

                             ('variation_dummies', ohe(),['variation'])

                            ],remainder='passthrough')
pipe= mp(transformer,dtc(random_state=42))

pipe
from sklearn.model_selection import train_test_split as tts
data.head()
x= data[['rating','variation','verified_reviews']].copy()

y= data.feedback

x.head()
x_train,x_test,y_train,y_test= tts(x,y,test_size=0.3,random_state=42,stratify=y)
x_train.shape,y_train.shape
pipe.fit(x_train,y_train)
pred=pipe.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,pred)                 #Accuracy of 100%
sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='.0f')
from sklearn.ensemble import RandomForestClassifier as rfc
pipe= mp(transformer, rfc(n_estimators=150, random_state=42))
pipe.fit(x_train,y_train)
pred=pipe.predict(x_test)
accuracy_score(y_test,pred)  # 99% accuracy
sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt='.0f')