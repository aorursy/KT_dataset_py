#coding:utf-8

#导入warnings包，利用过滤器来实现忽略警告语句。

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno
## 1) 载入训练集和测试集；

path = '../input/usedcarpred/'

Train_data = pd.read_csv(path+'used_car_train_20200313.csv', sep=' ')

Test_data = pd.read_csv(path+'used_car_testA_20200313.csv', sep=' ')
## 2) 简略观察数据(head()+shape)

Train_data.head().append(Train_data.tail())
Train_data.shape
Test_data.head().append(Test_data.tail())
Train_data.describe()
Test_data.describe()
## 2) 通过info()来熟悉数据类型

Train_data.info()
Test_data.info()
## 1) 查看每列的存在nan情况

Train_data.isnull().sum()
Test_data.isnull().sum()
# nan可视化

missing = Train_data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
# 可视化看下缺省值

msno.matrix(Train_data.sample(250))
msno.bar(Train_data.sample(1000))
Train_data.info()
Train_data['notRepairedDamage']
Train_data['notRepairedDamage'].value_counts()
Train_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
Train_data['notRepairedDamage'].value_counts()
Train_data.isnull().sum()
Test_data['notRepairedDamage'].value_counts()
Test_data['notRepairedDamage'].replace('-', np.nan, inplace=True)
Train_data["seller"].value_counts()
Train_data["offerType"].value_counts()
del Train_data["seller"]

del Train_data["offerType"]

del Test_data["seller"]

del Test_data["offerType"]
Train_data['price']