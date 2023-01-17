#let's import the library to read the dataset and couple of visualization library to analyze it further.



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import warnings 

warnings.filterwarnings('ignore')
data=pd.read_csv("../input/house-data/kc_house_data.csv")

data.head(n=5)
data.shape

data.info()
data.describe(include='all')
# Let's check for the null value 

data.isnull().sum()
# Heatmap would be the good choice to check the correlation betweeen the independent and the target variable.

plt.figure(figsize=(8,8))

sns.heatmap(data.corr())
fig, ax=plt.subplots(figsize=(12,4))

sns.boxplot(data['price'],showmeans=True)
data.corr().loc['price']
plt.figure(figsize=(5,5))

sns.jointplot(x='sqft_living',y='price',data=data,kind = 'reg')

sns.jointplot(x='yr_renovated',y='price',data=data)

sns.jointplot(x='sqft_lot',y='price',data=data,kind='reg')

sns.jointplot(x='floors',y='price',data=data,kind='reg')
sns.jointplot(x='sqft_above',y='price',data=data,kind='reg')

sns.jointplot(x='yr_built',y='price',data=data,kind='reg')
sns.jointplot(x='long',y='price',kind='reg',data=data)
sns.jointplot(x='sqft_lot15',y='price',data=data,kind = 'reg')

sns.jointplot(x='sqft_basement',y='price',data=data)

# Checking the number of Zeroes in the variable "sqft_basement" i.e house without basement it shows "13126" many zeros entries are there for the "sqft_basement'" variable.. 

data[(data['sqft_basement']==0)].count()['sqft_basement']
# Number of Zeros in the variable "sqft_basement" is "20699"..

data[data['yr_renovated']==0].count()['yr_renovated']
# Create two new columns for the analysis.

data['sqft_basement']=data['sqft_basement'].apply(lambda x: x if x>0 else None)

data['yr_renovated']=data['yr_renovated'].apply(lambda x: x if x>0 else None)

sns.jointplot(x='yr_renovated',y='price',data=data,dropna=True,kind='reg')

sns.jointplot(x='sqft_basement',y='price',data=data,kind='reg',dropna=True)



data[['sqft_basement','yr_renovated','price']].corr()
data['basement_present']=data['sqft_basement'].apply(lambda x: 1 if x>0 else 0)

data['basement_present']=data['basement_present'].astype('category')



data['renovated']=data['yr_renovated'].apply(lambda x: 1 if x>0 else 0)

data['renovated']=data['renovated'].astype('category')
g=sns.PairGrid(data, vars = ['sqft_living', 'sqft_living15', 'sqft_above'], size = 3.5)

g.map_upper(plt.scatter) 

g.map_diag(sns.distplot)

g.map_lower(plt.scatter, cmap="Blues_d")



data[['sqft_living', 'sqft_living15', 'sqft_above']].corr()
# Checking if variable "waterfront" has dependency on the house price
from scipy import stats

fig,ax=plt.subplots(figsize=(8,8))

sns.boxplot(x=data['waterfront'],y=data['price'],showmeans=True)

r, p = stats.pointbiserialr(data['waterfront'], data['price'])

print("pointbiserials r is {} and p is {}".format(r,p))
plt.subplots(figsize=(8,8))

sns.boxplot(x='renovated',y='price',data=data)

stats.pointbiserialr(data['renovated'],data['price'])
plt.subplots(figsize=(8,8))

sns.boxplot(x='basement_present',y='price',data=data)

stats.pointbiserialr(data['basement_present'],data['price'])
data.corr()['price']
data[['sqft_living','sqft_above','sqft_living15','price']].corr()
categorial_cols =['floors','view','condition','grade']

for variable in categorial_cols:

    dummies=pd.get_dummies(data[variable],drop_first=True)

    dummies = dummies.add_prefix("{}#".format(variable))

    data.drop(variable,axis=1,inplace=True)

    data=data.join(dummies)

data.drop(['sqft_basement','yr_renovated'],axis=1,inplace=True)

from sklearn.model_selection import train_test_split



x=data.drop(['id','date','price'],axis=1)

y=data['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LinearRegression

lm_obj=LinearRegression()

lm_obj.fit(x_train,y_train)
print("coefficents are:-{}".format(lm_obj.coef_))

print("intercept is:-{}".format(lm_obj.intercept_))
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,lm_obj.predict(x_test))
plt.plot(lm_obj.predict(x_test),y_test)
from sklearn.metrics import r2_score

r2score=r2_score(y_test,lm_obj.predict(x_test))

r2score


