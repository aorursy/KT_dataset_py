# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from pandas_profiling import ProfileReport

import tensorflow as tf

from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.shape
df.head(5)
df.describe()
df.isnull().sum().sum()
ProfileReport(df)
color = plt.cm.plasma

sns.heatmap(df.corr(), annot=True, cmap=color)
df.Age.max()
labels = ['Male', 'Female']

values = df.Sex.value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = df.Country.value_counts().index

values = df.Country.value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
labels = ['Dead', 'Survived']

values = df.Survived.value_counts().values



fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values)])

fig.show()
sns.countplot(x='Category', hue='Sex', data=df).set_title('Gender wise survivors distrbution')
male_survived = df['Sex'][(df['Sex']=='M') & (df['Survived']==1)].count()

female_survived = df['Sex'][(df['Sex']=='F') & (df['Survived']==1)].count()



male_all = df['Sex'][df['Sex']=='M'].count()

female_all =df['Sex'][df['Sex']=='F'].count()



perc_male = male_survived/male_all

perc_female = female_survived/female_all



print('Proportion of Male passengers that survived {:0.2f} '.format(perc_male*100))

print('Proportion of Female passengers that survived {:0.2f} '.format(perc_female*100))
sns.countplot(x='Category', hue='Survived', data=df).set_title('Passenger and Crew Survived Distribution')
df.drop(['Country', 'Firstname', 'Lastname'], axis=1, inplace=True)
df.head()
df.drop(['Category', 'Sex'],axis=1, inplace=True)

df.head()
df.isnull().sum()
x = df.drop(['Survived'], axis=1)

y = df['Survived']
sc=StandardScaler()

sc.fit(df.drop(['Survived', 'PassengerId'], axis = 1))

x_train = sc.transform(df.drop(['Survived', 'PassengerId'], axis = 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

preds = rfc.predict(x_test)

score = rfc.score(x_test,y_test)
score
preds[:10]
ground_truth = y_test[:10]

ground_truth
from sklearn.metrics import confusion_matrix, classification_report



cm = confusion_matrix(preds, y_test)

print('The Confusion Matrix : \n', cm)
sns.heatmap(cm, annot = True, cmap='coolwarm')
cf = classification_report(preds, y_test)

print('The Report : \n', cf)
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(x_train,y_train)
xgb_preds = xgb.predict(x_test)

xgb_score = xgb.score(x_test,y_test)

print('The Accuracy :',xgb_score)
cm = confusion_matrix(preds, y_test)

print('The Confusion Matrix : \n', cm)

sns.heatmap(cm, annot=True, cmap=color)
cf = classification_report(xgb_preds, y_test)

print('The Report : \n', cf)