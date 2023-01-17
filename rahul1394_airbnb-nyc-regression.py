import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

from sklearn.metrics import r2_score,mean_squared_error
airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb.head()
airbnb.info()
airbnb.isnull().sum()
airbnb.drop(['name','host_name'],axis=1,inplace=True)
airbnb[airbnb['reviews_per_month'].isna()]
airbnb['reviews_per_month'].fillna(0,inplace=True)
airbnb.drop('last_review',axis=1,inplace=True)
airbnb.tail()
sns.distplot(airbnb['price'])
sns.distplot(np.log1p(airbnb['price']))
airbnb['price'] = np.log1p(airbnb['price'])
sns.distplot(airbnb['minimum_nights'])
airbnb['minimum_nights'] = np.log1p(airbnb['minimum_nights'])
sns.distplot(airbnb['minimum_nights'])
plt.figure(figsize=(12,10))

sns.heatmap(airbnb.corr(),annot=True)
sns.countplot(airbnb['neighbourhood_group'])
sns.scatterplot(airbnb.latitude,airbnb.longitude)
sns.distplot(airbnb['number_of_reviews'])
sns.distplot(np.log1p(airbnb['number_of_reviews']))
sns.countplot(airbnb['room_type'])
airbnb['room_type'] = LabelEncoder().fit_transform(airbnb['room_type'])
airbnb['number_of_reviews'] = np.log1p(airbnb['number_of_reviews'])
airbnb.drop('calculated_host_listings_count',axis=1,inplace=True)
airbnb.drop(['id','host_id','reviews_per_month'],axis=1,inplace=True)
dummydata = pd.get_dummies(airbnb)
sc = StandardScaler()
scaledData = pd.DataFrame(sc.fit_transform(dummydata),columns=dummydata.columns)

scaledData.head()
x = scaledData.drop('price',axis=1)

y = scaledData['price']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30)
lr = LinearRegression()
lr.fit(xtrain,ytrain)

ypred = lr.predict(xtest)

ypred
r2_score(ytest,ypred)
np.sqrt(mean_squared_error(ytest,ypred))
r = Ridge()
r.fit(xtrain,ytrain)

ypredR = r.predict(xtest)
r2_score(ytest,ypredR)
np.sqrt(mean_squared_error(ytest,ypredR))
l = Lasso()
l.fit(xtrain,ytrain)
ypredLasso = l.predict(xtest)
r2_score(ytest,ypredLasso)
np.sqrt(mean_squared_error(ytest,ypredLasso))
e = ElasticNet()
e.fit(xtrain,ytrain)
ypredEnet = e.predict(xtest)
r2_score(ytest,ypredEnet)
np.sqrt(mean_squared_error(ytest,ypredEnet))
params = {

    'alpha' : [0.1,0.01,0.5,1,2,5,0.02,10,50],

    'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],

    'random_state' : [0,1,123,5,10]

}

params_lasso = {

    'alpha' : [0.1,0.01,0.5,1,2,5,0.02,10,50,25,100],

    'random_state' : [0,1,123,5,10]

}

params_enet = {

    'alpha' : [0.1,0.01,0.5,1,2,5,0.02,10,50,25,100],

    'random_state' : [0,1,123,5,10]

}
gridR = GridSearchCV(estimator=r,param_grid=params,cv =3)
#gridR.fit(xtrain,ytrain)