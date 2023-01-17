# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/column_2C_weka.csv')
df.columns

df.head()

df.info()

df.describe()
color_list = ['yellow' if i=='Abnormal' else 'red' for i in df.loc[:,'class']]

pd.plotting.scatter_matrix(df.loc[:, df.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = "d")

plt.show()
sns.countplot(x='class',data=df)

df.loc[:,'class'].value_counts()
df.corr()

ax=sns.heatmap(df.corr())
f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(),annot=True,linewidths=.3,fmt='.2f',ax=ax)

plt.show()
df['class']=[1 if each=='Abnormal' else 0 for each in df['class']]

df.sample(7)
y=df['class'].values

x_df=df.drop(['class'],axis=1)
x = (x_df - np.min(x_df))/(np.max(x_df)-np.min(x_df))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

Knn=KNeighborsClassifier(n_neighbors=3)

Knn.fit(x_train,y_train)

prediction=Knn.predict(x_test)

print(" {} nn score: {} ".format(3,Knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

Knn=KNeighborsClassifier(n_neighbors=13)

Knn.fit(x_train,y_train)

prediction=Knn.predict(x_test)

print(" {} nn score: {} ".format(13.5,Knn.score(x_test,y_test)))