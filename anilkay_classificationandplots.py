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
data=pd.read_csv("/kaggle/input/sloan-digital-sky-survey/Skyserver_SQL2_27_2018 6_51_39 PM.csv")

data.head()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
sns.countplot(data=data,x="class")
sns.countplot(data=data,x="rerun") 
data["rerun"].groupby(data["rerun"]).count()
del data["rerun"]
data.head()
data["run"].groupby(data["run"]).count()
data["camcol"].groupby(data["camcol"]).count()
data.describe()
data.shape
y=data["class"]
x=data[data.columns.difference(['class'])]
from sklearn.svm import SVC

svm=SVC(kernel="linear")

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_scale=sc.fit_transform(x)

from sklearn.model_selection import cross_val_score

cross_val_score(svm,x_scale,y,cv=5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.35, random_state=142)

svm=SVC(kernel="linear")

svm.fit(X_train,y_train)

ypred=ypred=svm.predict(X_test)

import sklearn.metrics as metr

print(metr.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metr.accuracy_score(y_pred=ypred,y_true=y_test))

print(metr.classification_report(y_pred=ypred,y_true=y_test))
from sklearn.manifold import TSNE

x_visu=TSNE(n_components=2).fit_transform(x_scale)
x_visuframe=pd.DataFrame(x_visu)

x_visuframe.columns=["bir","iki"]

x_visuframe.head()
sns.relplot(data=x_visuframe,x="bir",y="iki",hue=y)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

x_scale_lda=lda.fit_transform(x_scale,y)
x_visuframelda=pd.DataFrame(x_scale_lda)

x_visuframelda.columns=["bir","iki"]

x_visuframelda.head()
sns.relplot(data=x_visuframelda,x="bir",y="iki",hue=y)
lda = LinearDiscriminantAnalysis()

x_lda=lda.fit_transform(x,y)

x_ldaframe=pd.DataFrame(x_lda)

x_ldaframe.columns=["bir","iki"]

sns.relplot(data=x_ldaframe,x="bir",y="iki",hue=y)
correlation=x.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
xnew=x[x.columns.difference(['i','r'])]

xnew.head()
x.columns
correlation=xnew.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
xnew2=x[x.columns.difference(['mjd','g'])]

xnew2.head()
lda = LinearDiscriminantAnalysis()

x_lda=lda.fit_transform(xnew2,y)

x_ldaframe=pd.DataFrame(x_lda)

x_ldaframe.columns=["bir","iki"]

sns.relplot(data=x_ldaframe,x="bir",y="iki",hue=y)
xnew2
correlation=xnew2.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
xnew3=x[x.columns.difference(["z","i"])]

xnew3.head()
lda = LinearDiscriminantAnalysis()

x_lda=lda.fit_transform(xnew3,y)

x_ldaframe=pd.DataFrame(x_lda)

x_ldaframe.columns=["bir","iki"]

sns.relplot(data=x_ldaframe,x="bir",y="iki",hue=y)
correlation=xnew3.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
xnew4=xnew3[x.columns.difference(["z","i"])]

xnew3.head()
correlation=xnew4.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
xnew5=xnew4[xnew4.columns.difference(["mjd","r","plate"])]

correlation=xnew5.corr()

plt.figure(figsize=(16, 16))

sns.heatmap(correlation,annot=True)
lda = LinearDiscriminantAnalysis()

x_lda=lda.fit_transform(xnew5,y)

x_ldaframe=pd.DataFrame(x_lda)

x_ldaframe.columns=["bir","iki"]

sns.relplot(data=x_ldaframe,x="bir",y="iki",hue=y)