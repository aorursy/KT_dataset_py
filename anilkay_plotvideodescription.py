# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/youtubevideodataset/Youtube Video Dataset.csv")

data.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

sns.countplot(data=data,x="Category")
data2=data.dropna()
x=data2["Description"]

y=data2["Category"]



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer(stop_words="english",max_features=2000)

x_new=vectorizer.fit_transform(x)

x_new.shape
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

x_2_dimension=pca.fit_transform(x_new.toarray())
dataframe=pd.DataFrame(x_2_dimension)

dataframe.columns=["f1","f2"]
dataframe["Category"]=y

dataframe.head(6)
plt.figure(figsize=(18,18))

ax=sns.relplot(data=dataframe,x="f1",y="f2",hue="Category")

plt.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=10)

x_10_dimension=pca.fit_transform(x_new.toarray())



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



from sklearn.feature_selection import SelectKBest, chi2

X_new = SelectKBest(chi2, k=2).fit_transform(scaler.fit_transform(x_10_dimension), y)
dataframe=pd.DataFrame(X_new)

dataframe.columns=["f1","f2"]

dataframe["Category"]=y

plt.figure(figsize=(18,18))

ax=sns.relplot(data=dataframe,x="f1",y="f2",hue="Category")

plt.show()
pca = PCA(n_components=400)

x_400_dimension=pca.fit_transform(x_new.toarray())



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_400_dimension, y, test_size=0.33, random_state=421)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train=scaler.fit_transform(x_train)

X_test=scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=43)

knn.fit(X_train,y_train)

ypred=knn.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from sklearn.naive_bayes import GaussianNB

naive= GaussianNB()

naive.fit(X_train,y_train)

ypred=naive.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

lsvm=SVC(kernel="linear")

lsvm.fit(X_train,y_train)

ypred=lsvm.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from sklearn.svm import SVC

lsvm_pro=SVC(kernel="linear",probability=True)

lsvm_pro.fit(X_train,y_train)

ypred=lsvm_pro.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
from sklearn.tree import DecisionTreeClassifier

dtree= DecisionTreeClassifier(random_state=46)

dtree.fit(X_train,y_train)

ypred=dtree.predict(X_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
import shap

explainer=shap.TreeExplainer(dtree)

shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
sel= SelectKBest(chi2, k=10)

X_new_40010_train = sel.fit_transform(scaler.fit_transform(X_train), y_train)

X_new_40010_test=sel.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=43)

knn.fit(X_new_40010_train,y_train)

ypred=knn.predict(X_new_40010_test)

import sklearn.metrics as metrik

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))