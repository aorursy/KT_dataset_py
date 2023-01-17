# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt



df_sample = pd.read_csv("../input/sample_airbnb.csv")

df_train = pd.read_csv("../input/train_airbnb.csv")

df_test = pd.read_csv("../input/test_airbnb.csv")

sns.distplot(df_train[df_train['neighbourhood'] == 'Sydney']['price'])
for i in df_train['neighbourhood'].unique():

    print(i,' ',df_train[df_train['neighbourhood'] == i]['price'].sum())
for i in df_train['neighbourhood'].unique():

    print(i,' ',df_train[df_train['neighbourhood'] == i]['price'].mean())
for i in df_train['neighbourhood'].unique():

    print(i,' ',df_train[df_train['neighbourhood'] == i]['price'].max())
for i in df_train['room_type'].unique():

    print(i,' ',df_train[df_train['room_type'] == i]['price'].mean())
for i in df_train['room_type'].unique():

    print(i,' ',df_train[df_train['room_type'] == i]['price'].sum())
for i in df_train['room_type'].unique():

    print(i,' ',df_train[df_train['room_type'] == i]['price'].max())
buang = ['id','latitude','longitude', 'name','host_id','host_name','neighbourhood_group','minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

df_train.drop(buang, axis=1,inplace=True)

df_test.drop(buang, axis=1,inplace=True)
df_train.drop(df_train[df_train['price']==0].index,axis=0, inplace=True)
df_train['price'].describe()
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

df_combine = pd.concat([df_train,df_test],axis=0)

df_traincoba = pd.get_dummies(df_combine) #convert str to float

df_train_transform = df_traincoba[:len(df_train)]

df_test_transform= df_traincoba[len(df_train):]



X_train = df_train_transform.drop('price',axis=1)

Y_train = df_train_transform['price']





model = DecisionTreeRegressor()

model.fit(X_train, Y_train)

predict = model.predict(df_test_transform.drop('price',axis=1))



submit = pd.read_csv("../input/sample_airbnb.csv")

submit['price'] = predict

submit.to_csv("answer.csv",index=False)
