import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,PolynomialFeatures

from sklearn.linear_model import LinearRegression

%matplotlib inline
file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'

df=pd.read_csv(file_name)
df.head()
df.dtypes
df.describe()
df=df.drop(['Unnamed: 0','id'],axis=1)
df.describe()
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())

print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()

df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()

df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())

print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
df.floors.value_counts().to_frame()
sns.boxplot(x='waterfront',y='price',data=df)
df.waterfront.value_counts()
sns.regplot(x='sqft_above',y='price',data=df)
df.corr()['price'].sort_values()
X = df[['long']]

Y = df['price']

lm = LinearRegression()

lm.fit(X,Y)

lm.score(X, Y)
X1 = df[['sqft_living']]

Y1 = df['price']

lm = LinearRegression()

lm

lm.fit(X1,Y1)

lm.score(X1, Y1)
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     
X2 = df[features]

Y2 = df['price']

lm.fit(X2,Y2)

lm.score(X2,Y2)
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)

pipe

pipe.fit(X,Y)

pipe.score(X,Y)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

print("done")
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    

X = df[features]

Y = df['price']



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)





print("number of test samples:", x_test.shape[0])

print("number of training samples:",x_train.shape[0])
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge



RigeModel = Ridge(alpha=0.1) 

RigeModel.fit(x_train, y_train)

RigeModel.score(x_test, y_test)
pr=PolynomialFeatures(degree=2)

x_train_pr=pr.fit_transform(x_train[features])

x_test_pr=pr.fit_transform(x_test[features])



RigeModel = Ridge(alpha=0.1) 

RigeModel.fit(x_train_pr, y_train)

RigeModel.score(x_test_pr, y_test)