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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



corr = train.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print("Önemli özellikler belileniyor")

print(corr.SalePrice)
kategorik_özellikler = train.select_dtypes(include = ["object"]).columns

sayisal_özellikler = train.select_dtypes(exclude = ["object"]).columns

sayisal_özellikler = sayisal_özellikler.drop("SalePrice")

train_sayisal = train[sayisal_özellikler]

train_kategorik = train[kategorik_özellikler]
train_sayisal = train_sayisal.fillna(train_sayisal.median())

train_kategorik = pd.get_dummies(train_kategorik)
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression



train.SalePrice = np.log1p(train.SalePrice )

y = train.SalePrice



#sayisal ve kategorik veriler birleştiriliyor

train = pd.concat([train_kategorik,train_sayisal],axis=1)

train.shape



#Veri eğitim için bölünüyor. %70-%30 train-test olarak ayrılıyor

X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.3,random_state= 0)



#Linear regresyon ile test verisi eğitiliyor

lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)



#Elde edilen model ve gerçek değerin dağılımı görselleştirilmiştir

plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre, y_test, c = "yellow",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



print("Multiple linear regresyon : ",lr.score(X_test,y_test))