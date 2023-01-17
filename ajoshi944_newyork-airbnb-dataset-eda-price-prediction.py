# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

import warnings

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df['reviews_per_month'].fillna(value=0,inplace=True)
df.drop(['id','name','host_id','host_name','latitude','longitude','last_review','calculated_host_listings_count'],axis=1,inplace=True)
df.drop(df[df.price>1000].index,inplace=True)
df.drop(df[df.price==0].index,inplace=True)
sns.distplot(df['price'])
sns.barplot('neighbourhood_group','price',data=df)
sns.barplot('room_type','price',data=df)
fig=plt.figure(figsize=(12,16))

ax1=fig.add_subplot(411)

sns.scatterplot('minimum_nights','price',data=df,ax=ax1)

ax2=fig.add_subplot(412)

sns.scatterplot('number_of_reviews','price',data=df,ax=ax2)

ax3=fig.add_subplot(413)

sns.scatterplot('reviews_per_month','price',data=df,ax=ax3)

ax4=fig.add_subplot(414)

sns.scatterplot('availability_365','price',data=df,ax=ax4)

plt.show()
df.info()
categorical=['neighbourhood_group','neighbourhood','room_type']

#numeric=['id','host_id','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']

cat_df=pd.get_dummies(df[categorical])

#num_df=df[numeric].apply(pd.to_numeric)

df=pd.concat([df,cat_df],axis=1)
df.drop(['neighbourhood_group','neighbourhood','room_type'],axis=1,inplace=True)
df=df.drop_duplicates()
price=df['price']

df.drop('price',axis=1,inplace=True)
pca_df=PCA(n_components=15).fit_transform(df)
ny_df=pd.DataFrame(pca_df,columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12','feature13','feature14','feature15'])
ny_train,ny_test,price_train,price_test = train_test_split(ny_df,price,test_size=0.33)
ny_lr=LinearRegression().fit(ny_train,price_train)

ny_logr=LogisticRegression().fit(ny_train,price_train)

ny_svm=svm.SVR().fit(ny_train,price_train)

ny_rf=RandomForestRegressor().fit(ny_train,price_train)



print('-----------Scores-------------')

print('Linear Regression:{}\n'.format(ny_lr.score(ny_test,price_test)))

print('Logistic Regression:{}\n'.format(ny_logr.score(ny_test,price_test)))

print('SVM:{}\n'.format(ny_svm.score(ny_test,price_test)))

print('Random Forest:{}'.format(ny_rf.score(ny_test,price_test)))