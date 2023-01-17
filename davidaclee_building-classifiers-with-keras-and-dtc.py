# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from random import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')
df.head(10)
df = df.replace('?', np.nan)
df.isnull().sum()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
print("字段数据最大值大于 1 的列数量有：", len(df.describe().loc['max', :][(df.describe().loc['max', :]>1)]))
print("字段数据最大值等于 1 的列数量有：", len(df.describe().loc['max', :][(df.describe().loc['max', :]==1)]))
print("字段数据最大值等于 0 的列数量有：", len(df.describe().loc['max', :][(df.describe().loc['max', :]==0)]))
# print(df.describe().loc['max', :][(df.describe().loc['max', :]==0)].index.tolist())
# 输出 ['STDs:cervical condylomatosis', 'STDs:AIDS']，可以进行 drop() 处理
max_zero = df.describe().loc['max', :][(df.describe().loc['max', :]==0)].index.tolist()
df = df.drop(max_zero, axis=1)

max_one = df.describe().loc['max', :][(df.describe().loc['max', :]==1)].index.tolist()
for col in max_one:
    df[col] = df[col].fillna(random.choice([0, 1]))
    
max_more = df.describe().loc['max', :][(df.describe().loc['max', :]>1)].index.tolist()
for col in max_more:
    df[col] = df[col].fillna(df[col].median())

df_corr = df
df = df.as_matrix()
shuffle(df)
train = df[:int(len(df)*0.8), :]
test = df[int(len(df)*0.8):, :]
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
train = minmax_scale.fit_transform(train)
test = minmax_scale.fit_transform(test)
net = Sequential()
net.add(Dense(input_dim = 33, units = 6))
net.add(Activation('relu'))
net.add(Dense(units=1))
net.add(Activation('sigmoid'))
net.compile(loss='binary_crossentropy', optimizer='adam')
net.fit(train[:,:-1], train[:,-1], epochs=50, batch_size=20)
predict_result = net.predict_classes(train[:, :-1]).reshape(len(train))
cm = confusion_matrix(train[:,-1], predict_result) #混淆矩阵
# net.predict_classes(train[:,:-1])

plt.matshow(cm, cmap=plt.cm.Greens)
plt.colorbar()

for x in range(len(cm)):
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
        
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
classifier = DecisionTreeClassifier()
classifier.fit(train[:, :-1], train[:, -1])
cm = confusion_matrix(train[:,-1], classifier.predict(train[:,:-1]))
plt.matshow(cm, cmap=plt.cm.Greens) 
plt.colorbar()

for x in range(len(cm)):
  for y in range(len(cm)):
    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
corrmat = df_corr.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=2, square=True, cmap='rainbow')
k = 10 
cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index
cm = np.corrcoef(df_corr[cols].values.T)

plt.figure(figsize=(9, 9))

sns.set(font_scale=1.25)

hm=sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                annot_kws={'size':10}, yticklabels = cols.values, xticklabels = cols.values)
