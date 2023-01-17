import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")
train.head()
train.info()
train.shape
train.describe()
train.isnull().sum()
train
train['Date']=pd.to_datetime(train['Date'])
train.info()
train['Year']=train['Date'].dt.year

train['Month']=train['Date'].dt.month

train['Day']=train['Date'].dt.day
train.tail()
train['Year'].value_counts()
train.hist(figsize=(12,12))
train.Open[train.Year==2012].plot(kind='kde')

train.Open[train.Year==2013].plot(kind='kde')

train.Open[train.Year==2014].plot(kind='kde')

train.Open[train.Year==2015].plot(kind='kde')

train.Open[train.Year==2016].plot(kind='kde')
train.Volume[train.Year==2012].plot(kind='kde')

train.Volume[train.Year==2013].plot(kind='kde')

train.Volume[train.Year==2014].plot(kind='kde')

train.Volume[train.Year==2015].plot(kind='kde')

train.Volume[train.Year==2016].plot(kind='kde')
g=sns.FacetGrid(train,col='Year',row='Month')

g.map(plt.hist,'Open')
corr=train.corr()
sns.heatmap(corr,annot=True)