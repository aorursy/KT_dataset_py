import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import requests

import os

df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')

df.head()
df=df.drop(columns='App')

df=df.drop(index=10472)

df.head()
df.Reviews=df.Reviews.apply(lambda r: int(r))

type(df.Reviews[0])
df.Size=df.Size.apply(lambda x: float(x[:len(x)-1])*10**6 if 'M' in x else float(x[:len(x)-1])*10**3 if 'k' in x else np.nan)

type(df.Size[0])
import re

df.Installs=df.Installs.apply(lambda x:int(re.sub('[+,,]','',x)))

type(df.Installs[0])
df.Type.unique()
df.Price=df.Price.apply(lambda x: float(re.split('\$',x)[1]) if '$' in x else 0.0)
column_name={"Content Rating":"Content_rating","Last Updated":"Last_update","Current Ver":"Current_ver","Android Ver":"Android_ver"}

df=df.rename(columns=column_name)

df.columns.values
f,ax=plt.subplots(figsize=(10,8))

sns.countplot('Category',data=df,ax=ax)

plt.xticks(rotation='vertical')
df.Category.value_counts()
f,ax=plt.subplots(figsize=(10,8))

plt.scatter(x=df.Category,y=df.Rating)

plt.xticks(rotation='vertical')

plt.ylim(0,6)
rating=df.groupby(['Category']).median()

f,ax=plt.subplots(figsize=(10,8))

plt.scatter(x=rating.index,y=rating.Rating.values)

plt.xticks(rotation='vertical')

plt.ylim(3,6)
rating.Rating[rating.Rating==rating.Rating.min()]

rating.Rating[rating.Rating==rating.Rating.max()]
f,ax=plt.subplots(figsize=(10,8))

sns.countplot(df.Category[df.Rating.isnull()])

plt.xticks(rotation='vertical')
f,ax=plt.subplots(figsize=(10,8))

plt.scatter(df.Category,df.Size)

plt.xticks(rotation='vertical')
df.Category[df.Size==df.Size.max()]

df.Category[df.Size==df.Size.min()]
install=df.groupby('Category').sum()

f,ax=plt.subplots(figsize=(10,8))

plt.bar(x=install.index.values,height=install.Installs)

plt.xticks(rotation='vertical')
install[install.Installs==install.Installs.max()]

install[install.Installs==install.Installs.min()]
df2=df.groupby('Type').count()

f,ax=plt.subplots(figsize=(5,5))

plt.pie(df2.Category,labels=df2.index.values,autopct='%1.1f%%')
f,ax=plt.subplots(figsize=(10,8))

plt.bar(x=df2.index.values,height=df2.Category)

for i,j in zip(df2.index.values,df2.Category): 

    plt.text(i, j, str(j))
f,ax=plt.subplots(figsize=(10,8))

sns.countplot('Content_rating',data=df,ax=ax)

plt.xticks(rotation='vertical')

content=df.groupby('Content_rating').count()

content[content.Category==content.Category.max()]

content[content.Category==content.Category.min()]
f,ax=plt.subplots(figsize=(10,8))

sns.countplot('Category',data=df,ax=ax,hue='Content_rating')

plt.xticks(rotation='vertical')
f,ax=plt.subplots(figsize=(10,8))

sns.catplot('Content_rating','Price',data=df,ax=ax)

#sns.catplot('Content_rating','Installs',data=df)

plt.xticks(rotation='vertical')
sns.relplot('Category','Price',data=df,col='Content_rating',col_wrap=3)
sns.relplot('Category','Rating',data=df,col='Content_rating',col_wrap=3)
df['category_num']=df.Category.replace(list(df.Category.unique()),range(0,len(list(df.Category.unique()))))

df['type_num']=df.Type.replace(['Free','Paid'],range(0,2))

df['content_rating_num']=df.Content_rating.replace(list(df.Content_rating.unique()),range(0,len(list(df.Content_rating.unique()))))

df['installs_num']=df.Installs.apply(lambda x: 2 if x<=1000000000 and x>=500000000 else 1 if x<500000000 and x>=15464338 else 0)

df=df.dropna(axis='rows')

x=df.loc[:,['Rating','Size','category_num','type_num','content_rating_num']]

y=df.installs_num
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.svm import SVC

svc=SVC().fit(x_train,y_train)

print(svc.score(x_test,y_test))

print(svc.score(x_train,y_train))
from sklearn.linear_model import LogisticRegression

log=LogisticRegression().fit(x_train,y_train)

print(log.score(x_train,y_train),log.score(x_test,y_test))
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=3).fit(x_train,y_train)

print(knn.score(x_train,y_train),knn.score(x_test,y_test))