# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df
df.columns
df.describe().T
df.isnull().sum()
df['last_review']
df['reviews_per_month']
#lets drop some irrelevant features from the dataset
df=df.drop(['name','host_name','last_review','reviews_per_month','id','host_id','name','neighbourhood'],axis=1)
df
df['neighbourhood_group'].unique()
import matplotlib.pyplot as plt

import seaborn as sns
plt.scatter(df['number_of_reviews'],df['price'])

plt.show()

#here we can see that price decreases as the no of reviews increased
plt.scatter(df['minimum_nights'],df['price'])

plt.show()

#no clear pattern visible here
plt.plot(df['calculated_host_listings_count'],df['price'])

plt.show()

#no fixed pattern here too
plt.plot(df['availability_365'],df['price'])

plt.show()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
df
sns.heatmap(df.isnull())
y=df['price']

x=df.drop(['price'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr=LinearRegression()
lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)
score_1=r2_score(y_test,pred_1)
score_1
pred_1
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)

pred_2=rfr.predict(x_test)

score_2=r2_score(y_test,pred_2)
score_2
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor()

gbr.fit(x_train,y_train)

pred_3=gbr.predict(x_test)

score_3=r2_score(y_test,pred_3)
score_3