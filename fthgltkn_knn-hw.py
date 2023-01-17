# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1=pd.read_csv('../input/column_2C_weka.csv')

data2=pd.read_csv('../input/column_3C_weka.csv')
data1.info()

data1.isnull().sum()

data1.head()
data_1=(data1['class']).unique()

sns.barplot(x=data_1,y=data1['class'].value_counts())

plt.xlabel('class')

plt.ylabel('values')

plt.grid()

plt.show()
data2.info()

data2.isnull().sum()

data2.head()
data_2=(data2['class']).unique()

sns.barplot(x=data_2,y=data2['class'].value_counts())

plt.xlabel('class')

plt.ylabel('values')

plt.grid()

plt.show()
data1.head()
data1['class']=pd.DataFrame(1 if each=='Normal' else 0 for each in data1['class'])

x_1=data1['class']

y_1=data1.drop(['class'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(y_1,x_1,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

print('{} nn değeri için {}'.format(3,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()


