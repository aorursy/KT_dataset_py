#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#load dataset

df = pd.read_csv('../input/breast-cancer-data/data.csv')

df.head()
df.info()
df.isna().sum()
df= df.dropna(axis=1)
df.info()
#count of malignant and benignant

df['diagnosis'].value_counts()
sns.countplot(df['diagnosis'], label = "count")
df.dtypes
#encoding categorical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.iloc[:, 1]= le.fit_transform(df.iloc[:,1].values)
print(df.iloc[:, 1])
df.head()
df.corr()
#heatmap

plt.figure(figsize= (20,20))

sns.heatmap(df.corr(), annot = True, fmt= '.0%')
# train test split

from sklearn.model_selection import train_test_split

X = df.drop(['diagnosis'], axis=1)

Y = df.diagnosis.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#support vector classifier

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train, y_train)

print("SVC accuracy : {:.2f}%".format(svm.score(X_test, y_test)*100))
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

print("Naive Bayes accuracy : {:.2f}%".format(nb.score(X_test, y_test)*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 1000, random_state = 1)

rf.fit(X_train, y_train)

print("Random Forest accuracy : {:.2f}%".format(rf.score(X_test, y_test)*100))
import xgboost

xg = xgboost.XGBClassifier()

xg.fit(X_train, y_train)

print("XG boost accuracy : {:.2f}%".format(xg.score(X_test, y_test)*100))
#SVC accuracy : 58.77%

#Naive Bayes accuracy : 59.65%

#Random Forest accuracy : 95.61%

#XG boost accuracy : 97.37%
