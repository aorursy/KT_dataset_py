# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt;

from sklearn import linear_model,metrics

from sklearn import model_selection

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

%matplotlib inline

#print(os.listdir("../input/sample_submission.csv"))

data=pd.read_csv("../input/sample_submission.csv")

df_train = pd.read_csv("../input/train.csv")

df_test= pd.read_csv("../input/test.csv")

print(data.head(5))

print(df_train.head(5))

print(df_test.head(5))







# Any results you write to the current directory are saved as output.
#print(df_train.columns)

plt.disconnect
X=df_train[['popularity','budget','id']]

Y=df_train['revenue']

regression=linear_model.LinearRegression()

regression.fit(X,Y)
X_Test=df_test[['popularity','budget','id']]

pred=regression.predict(X_Test)

results = model_selection.cross_val_score(regression, X_Test,pred)
print('Variance score: {}'.format(regression.score(X_Test,pred)))
print(df_train.columns)
sns.distplot(df_train['revenue'],kde=False,bins=10)
#sns.jointplot(x=df_train['id'],y=df_train['revenue'],data=df_train)

sns.rugplot(df_train['revenue'])
sns.barplot(x='original_language',y='revenue',data=df_train)
sns.boxplot(x='title',y='revenue',data=df_train)
tc=df_train.corr()
sns.heatmap(tc,annot=True,linewidths=1)
g=sns.PairGrid(df_train)

g.map(plt.scatter)