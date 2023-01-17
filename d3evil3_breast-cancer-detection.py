# importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# load dataset

df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.info()
df.isna().sum()
df = df.dropna(axis=1)
df.info()
# count of malignant and benignate

df['diagnosis'].value_counts()
sns.countplot(df['diagnosis'], label = 'count')
df.dtypes
# encoding Categorical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.iloc[:,1] = le.fit_transform(df.iloc[:,1].values)
df.head()
data_mean=df.iloc[:,1:11]
#Plot histograms of CUT1 variables

hist_mean=data_mean.hist(bins=10, figsize=(15, 10),grid=False,)
#Heatmap

plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),annot=True, fmt = '.0%')
#Density Plots

plt = data_mean.plot(kind= 'density', subplots=True, layout=(4,3), sharex=False, 

                     sharey=False,fontsize=12, figsize=(15,10))
# train test split

from sklearn.model_selection import train_test_split
x = df.drop(['diagnosis'], axis=1)

y = df['diagnosis'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(x_train,y_train)

print("Logistic Regression accuracy : {:.2f}%".format(reg.score(x_test,y_test)*100))
# support vector classifier

from sklearn.svm import SVC

svm = SVC(random_state=1)



svm.fit(x_train,y_train)

print("SVC accuracy : {:.2f}%".format(svm.score(x_test,y_test)*100))
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print(" Naive Bayes accuracy : {:.2f}%".format(nb.score(x_test,y_test)*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000,random_state=1)

rf.fit(x_train,y_train)

print("Random Forest Classifier accuracy : {:.2f}%".format(rf.score(x_test,y_test)*100))
import xgboost

xg = xgboost.XGBClassifier()

xg.fit(x_train,y_train)

print("XGboost accuracy : {:.2f}%".format(xg.score(x_test,y_test)*100))