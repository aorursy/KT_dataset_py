import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline
file_name = '../input/predicting-house-pricehouse-sales-in-king-county/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
df.head()
df.dtypes
df.describe()
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df.describe()
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())
unique_floor = df['floors'].value_counts()
unique_floor.to_frame()
sns.boxplot(x="waterfront", y="price", data=df)
sns.regplot(x="sqft_above", y="price", data=df)
df.corr()['price'].sort_values()
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)
x = df[['sqft_living']]
y = df['price']
lm.fit(x,y)
print(lm.score(x,y))
yhat = lm.predict(x)
print(yhat)
features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]
lm.fit(features,df['price'])
print(lm.score(features,y))
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(features,y)
print(pipe.score(features,y))
ypipe=pipe.predict(features)
print(ypipe)
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
Ridge_Model = Ridge(alpha=0.1)
Ridge_Model.fit(x_train, y_train)
print(Ridge_Model.score(x_test, y_test))
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

RigeModel = Ridge(alpha=0.1) 
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)
