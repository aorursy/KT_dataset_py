import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import itertools

import sklearn as sk

import warnings
sns.set(style='white', context='notebook', palette='deep')

import matplotlib.style as style

style.use('fivethirtyeight')
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
df=pd.read_csv('../input/nyc-rolling-sales.csv')
df.head(5)
df.drop(columns={'Unnamed: 0', 'EASE-MENT'}, inplace=True)
df.head()
df.columns
df.dtypes
df['BOROUGH'][df['BOROUGH']==1]='Manhattan'
df['BOROUGH'][df['BOROUGH']==2]='Bronx'
df['BOROUGH'][df['BOROUGH']==3]='Brooklyn'
df['BOROUGH'][df['BOROUGH']==4]='Queens'
df['BOROUGH'][df['BOROUGH']==5]='Staten Island'
df.head()
df.info()
missing_data=df.isnull()
missing_data.head()
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")
sum(df.duplicated(df.columns))
df=df.drop_duplicates(df.columns, keep='last')
sum(df.duplicated(df.columns))
df.dtypes
df.head(2)
df['SALE PRICE']=pd.to_numeric(df['SALE PRICE'], errors='coerce')
df['YEAR BUILT']=pd.to_numeric(df['YEAR BUILT'], errors='coerce')
df['LAND SQUARE FEET']=pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
df['GROSS SQUARE FEET']=pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
df['SALE DATE']=pd.to_datetime(df['SALE DATE'], errors='coerce')
df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')

df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')

df['ZIP CODE'] = df['ZIP CODE'].astype('category')
df.head()
df.isnull().sum()
avg_land=df['LAND SQUARE FEET'].astype('float').mean(axis=0)
df['LAND SQUARE FEET'].replace(np.nan, avg_land, inplace=True)
avg_gross=df['GROSS SQUARE FEET'].astype('float').mean(axis=0)
df['GROSS SQUARE FEET'].replace(np.nan, avg_gross, inplace=True)
avg_sale_price=df['SALE PRICE'].astype('float').mean(axis=0)
df['SALE PRICE'].replace('np.nan, avg_sale_price', inplace=True)
df.head()
var=df.columns

count=[]

for variable in var:

    length=df[variable].count()

    count.append(length)
plt.figure(figsize=(40,8))

sns.barplot(x=var, y=count)

plt.title('Percentage of Available Data', fontsize=30)

plt.show()
df.corr()
df[['COMMERCIAL UNITS', 'SALE PRICE']].corr()
sns.regplot(x='COMMERCIAL UNITS', y='SALE PRICE', data=df)
df[['LAND SQUARE FEET', 'SALE PRICE']].corr()
sns.regplot(x='LAND SQUARE FEET', y='SALE PRICE', data=df)
df[['GROSS SQUARE FEET', 'SALE PRICE']].corr()
sns.regplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=df)
sns.boxplot(x='BOROUGH', y='SALE PRICE', data=df)
df.head(2)
df.describe(include=['object'])
df_borough=df[['BOROUGH','SALE PRICE', 'SALE DATE']]
df_borough.head(5)
df_borough=df_borough.groupby(['BOROUGH'], as_index=False).mean()
df_borough.head(100)
sns.boxplot(x='BOROUGH', y='SALE PRICE', data=df_borough)
df_manhattan=df[(df['BOROUGH']=='Manhattan')]
df_manhattan
df_manhattan_neighborhood=df[['NEIGHBORHOOD', 'RESIDENTIAL UNITS','SALE PRICE', 'SALE DATE']]
df_manhattan_neighborhood=df_manhattan_neighborhood.groupby(['NEIGHBORHOOD', 'SALE PRICE'], as_index=False).mean()
df_manhattan_neighborhood
fig, ax = plt.subplots(figsize=(40,20))

plt.xticks(fontsize=30) 

plt.yticks(fontsize=30)

ax.set_title('Neighborhood Sale Price Analysis', fontweight="bold", size=30)

ax.set_ylabel('Neighborhood', fontsize = 30)

ax.set_xlabel('Sale Price', fontsize = 30)

sns.boxplot(x='SALE PRICE', y='NEIGHBORHOOD', data=df_manhattan)
df_manhattan.head(2)
fig, ax = plt.subplots(figsize=(40,20))

plt.xticks(fontsize=30) 

plt.yticks(fontsize=30)

ax.set_title('Neighborhood vs Residential Units Analysis', fontweight="bold", size=30)

ax.set_ylabel('Neighborhood', fontsize = 30)

ax.set_xlabel('Residential Units', fontsize = 30)

sns.barplot(x='RESIDENTIAL UNITS', y='NEIGHBORHOOD', data=df_manhattan)
plt.figure(figsize=(12,4))

sns.countplot(x='BOROUGH', data=df)
def get_season(x):

    if x==1:

        return 'Summer'

    elif x==2:

        return 'Fall'

    elif x==3:

        return 'Winter'

    elif x==4:

        return 'Spring'

    else:

        return ''

df['seasons']=df['SALE DATE'].apply(lambda x:x.month)

df['seasons']=df['seasons'].apply(lambda x:(x%12+3)//3)

df['seasons']=df['seasons'].apply(get_season)
plt.figure(figsize=(18,8))

df_wo_manhattan=df.loc[df['BOROUGH']!='Manhattan']

sns.relplot(x="BOROUGH", y="SALE PRICE",hue='seasons' ,kind="line", data=df_wo_manhattan,legend='full')
sns.regplot(x='SALE PRICE', y='LAND SQUARE FEET', data=df)
from sklearn.linear_model import LinearRegression
df[['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET']]
df.dropna(subset=["SALE PRICE"], axis=0, inplace = True)
df.reset_index(drop = True, inplace = True)
df.dropna(subset=['GROSS SQUARE FEET', 'LAND SQUARE FEET'], axis=0, inplace=True)
lm = LinearRegression()

lm
X = df[['SALE PRICE']]

Y = df['GROSS SQUARE FEET']
lm.fit(X,Y)
Yhat=lm.predict(X)

Yhat[0:5]   
## intercept value is

lm.intercept_
## slope

lm.coef_
plt.figure(figsize=(12, 10))

sns.regplot(x="GROSS SQUARE FEET", y="SALE PRICE", data=df)

plt.ylim(0,)
sns.pairplot(data=df, hue='BOROUGH')
variable_model=['BOROUGH','BUILDING CLASS CATEGORY','COMMERCIAL UNITS','GROSS SQUARE FEET',

               'SALE PRICE','Building Age During Sale','LAND SQUARE FEET','RESIDENTIAL UNITS','seasons']

data_model=df.loc[:,variable_model]
important_features=['BOROUGH','BUILDING CLASS CATEGORY','seasons']

longest_str=max(important_features,key=len)

total_num_of_unique_cat=0

for feature in important_features:

    num_unique=len(data_model[feature].unique())

    print('{} : {} unique categorical values '.format(feature,num_unique))

    total_num_of_unique_cat+=num_unique

print('Total {} will be added with important feature adding'.format(total_num_of_unique_cat))
df[df['SALE PRICE']==0.0].sum().count()
important_features_included = pd.get_dummies(data_model[important_features])

important_features_included.info(verbose=True, memory_usage=True, null_counts=True)
data_model.drop(important_features,axis=1,inplace=True)

data_model=pd.concat([data_model,important_features_included],axis=1)

data_model.head()
plt.figure(figsize=(12,8))

sns.distplot(data_model['SALE PRICE'],bins=2)

plt.title('Histogram of SALE PRICE')

plt.show()
data_model.head()
data_model=data_model[data_model['SALE PRICE']!=0]

data_model
data_model['SALE PRICE'] = StandardScaler().fit_transform(np.log(data_model['SALE PRICE']).values.reshape(-1,1))

plt.figure(figsize=(10,6))

sns.distplot(data_model['SALE PRICE'])

plt.title('Histogram of Normalised SALE PRICE')

plt.show()
data_model.describe()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
lm.fit(X, Y)
lm.score(X, Y)
Yhat=lm.predict(X)

Yhat[0:4]
from sklearn.metrics import mean_squared_error
mean_squared_error(df['SALE PRICE'], Yhat)
y=data_model['SALE PRICE']

X=data_model.drop('SALE PRICE',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

print('Size of Training data: {} \n Size of test data: {}'.format(X_train.shape[0],X_test.shape[0]))
data_model.shape[0]
sns.distplot(y_test)

plt.show()