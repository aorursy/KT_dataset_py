# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")
df.head()
df.info()
df["Time"].describe()
df["Time"].unique()  # time in seconds since the first transaction in dataset
df["Class"].unique()
reals = df[df["Class"]==0];
frauds = df[df["Class"]==1];
# randomly sample some fields and see what they look like
rand_reals_v1 = reals["V1"].sample(n=100);
rand_fakes_v1 = frauds["V1"].sample(n=100);
plt.subplot(1,2,1);
plt.scatter(x=np.linspace(1,100,100),y=rand_reals_v1);
plt.subplot(1,2,2);
plt.scatter(x=np.linspace(1,100,100),y=rand_fakes_v1);
rand_reals_v1.describe()
rand_fakes_v1.describe()
df2 = df[["V1", "V2", "V3", "V4", "V5"]].sample(n=1000)
pd.plotting.scatter_matrix(df2,figsize=(12,12))
groups = df.groupby(df["Class"])
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V2, marker='o', linestyle='', ms=12, label=name)
ax.legend()
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V3, marker='o', linestyle='', ms=12, label=name)
ax.legend()
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V4, marker='o', linestyle='', ms=12, label=name)
ax.legend()
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V5, marker='o', linestyle='', ms=12, label=name)
ax.legend()
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V6, marker='o', linestyle='', ms=12, label=name)
ax.legend()
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.V1, group.V7, marker='o', linestyle='', ms=12, label=name)
ax.legend()
import sklearn.utils
df[df["Class"]==1]["V1"].count()
df[df["Class"]==0]["V1"].count()
# try naive upsampling (probably not the wisest)
nToSample = df[df["Class"]==0]["V1"].count() - df[df["Class"]==1]["V1"].count();
frauds = sklearn.utils.resample(df[df["Class"]==1],n_samples=nToSample)
frauds["V1"].count()
reals = df[df["Class"]==0]
reals["V1"].count()
df_upsampled = pd.concat([reals, frauds])
df_upsampled.head()
# verify we upsampled well
print(df_upsampled[df_upsampled["Class"]==1]["V1"].count())
print(df_upsampled[df_upsampled["Class"]==0]["V1"].count())
import sklearn.model_selection
del df_upsampled["Time"]
df_upsampled.head()
df_upsampled.iloc[0:2, 0:27]
df_model_train = df_upsampled.sample(frac=0.1)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(df_model_train.iloc[:,0:27],df_model_train["Class"],test_size=0.2)
SVC = sklearn.svm.LinearSVC()
SVC.fit(X_train,Y_train)
SVC.score(X_test,Y_test)
from sklearn.model_selection import GridSearchCV
param_grid = {'penalty': ['l2'],
              'loss':['hinge','squared_hinge'],
              'C':[1, 10, 100, 1000]}
SVC2 = sklearn.svm.LinearSVC()
clf = GridSearchCV(SVC2,param_grid)
clf.fit(X_train,Y_train)
clf.score(X_test, Y_test)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X_train, Y_train)
scores.mean()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
