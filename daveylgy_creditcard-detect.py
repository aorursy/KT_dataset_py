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
# 神经网络

from keras.layers import Dense,Dropout,Reshape,Flatten

from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler# 最小值最大值归一化

from sklearn.model_selection import train_test_split # 训练测试分割

import pandas as pd

import numpy as np

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

# 拆分为特征矩阵和目标向量

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

X = X.values

Y = Y.values

# 归一化

s = MinMaxScaler()

s.fit(X)

X = s.transform(X)

# 分割训练集和测试集

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

# hidden_layer_sizes接收的是一个元组，以元组内元素的个数，代表神经网络的层数，以元素的值代表神经元的个数

#

model = Sequential()

model.add(Dense(units=10,activation='relu'))

# model.add(Dropout(rate=0.6))

model.add(Dense(units=10,activation='relu'))

# model.add(Dropout(rate=0.6))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')

# MLP = MLPClassifier(hidden_layer_sizes=(5, 5, 5, 3), activation='identity', solver='adam')

model.fit(train_x, train_y)

print('神经网络:',model.evaluate(x=test_x,y=test_y))
# KNN

from sklearn.preprocessing import MinMaxScaler# 最小值最大值归一化

from sklearn.model_selection import train_test_split # 训练测试分割

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

import pandas as pd

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

# 拆分为特征矩阵和目标向量

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

# 归一化

s = MinMaxScaler()

s.fit(X)

X = s.transform(X)

# 分割训练集和测试集

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.1)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(train_x,train_y)

print('KNN:',knn.score(test_x, test_y))
# 线性回归

from sklearn.preprocessing import MinMaxScaler# 最小值最大值归一化

from sklearn.model_selection import train_test_split # 训练测试分割

from sklearn.linear_model import LarsCV,LassoCV

import pandas as pd



# 拆分为特征矩阵和目标向量

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

# 归一化

s = MinMaxScaler()

s.fit(X)

X = s.transform(X)

# 分割训练集和测试集

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.1)

# LarsCV

liner = LarsCV()

liner.fit(train_x,train_y)

print('LarsCV:',liner.score(test_x, test_y))

# LassoCV

liner = LassoCV()

liner.fit(train_x,train_y)

print('LassoCV:',liner.score(test_x, test_y))
# 逻辑回归

from sklearn.preprocessing import MinMaxScaler# 最小值最大值归一化

from sklearn.model_selection import train_test_split # 训练测试分割

from sklearn.linear_model import LogisticRegression

import pandas as pd



# 拆分为特征矩阵和目标向量

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

# 归一化

s = MinMaxScaler()

s.fit(X)

# print(s.transform([[26575,10.650102,0.866627]]))

# 分割训练集和测试集

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.1)

logic = LogisticRegression(max_iter=1000)

logic.fit(train_x,train_y)

print('LogisticRegression:',logic.score(test_x, test_y))
# 决策树

from sklearn.preprocessing import MinMaxScaler# 最小值最大值归一化

from sklearn.model_selection import train_test_split # 训练测试分割

from sklearn.tree import DecisionTreeClassifier

import pandas as pd



# 拆分为特征矩阵和目标向量

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

# 归一化

s = MinMaxScaler()

s.fit(X)

# print(s.transform([[26575,10.650102,0.866627]]))

# 分割训练集和测试集

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.1)

Dtree = DecisionTreeClassifier()

Dtree.fit(train_x,train_y)

print('DecisionTreeClassifier:',Dtree.score(test_x, test_y))
# SVM

from sklearn.preprocessing import MinMaxScaler# 最小值最大值归一化

from sklearn.model_selection import train_test_split # 训练测试分割

from sklearn.svm import SVC



import pandas as pd

# 拆分为特征矩阵和目标向量

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

# 归一化

s = MinMaxScaler()

s.fit(X)

print(s.transform([[26575,10.650102,0.866627]]))

# 分割训练集和测试集

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.1)

SVM = SVC(C=100,max_iter=-1)

SVM.fit(train_x, train_y)

print('SVM:',SVM.score(test_x, test_y))