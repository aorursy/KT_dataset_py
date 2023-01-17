# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head()
df.info()
df.describe()
df.corr()['price'].sort_values()[:-1]
df.drop('id',axis=1,inplace=True)
df.select_dtypes(object).columns
df['date']
df['year'] = df['date'].apply(lambda x: int(x[:4]))

df['month'] = df['date'].apply(lambda x: int(x[4:6]))

df['day'] = df['date'].apply(lambda x: int(x[6:8]))
df.head()
df.drop('date',axis=1,inplace=True)
df.info()
df['bedrooms']
df['bedrooms'].nunique()
df['bedrooms'].value_counts()
sns.scatterplot(x='bedrooms',y='price',data=df)
df['bedrooms']=df['bedrooms'].replace([33, 11,10,9,8], 7)
df.groupby('bedrooms')['price'].mean()
df.groupby('bedrooms')['price'].std()
df['bedrooms'].mean()
df['bedrooms'].std()
df.corr()['bedrooms']['price']
df.head()
df['bathrooms']
df['bathrooms'].nunique()
sns.distplot(df['bathrooms'])
plt.figure(figsize=(12,6))

sns.barplot(x='bathrooms',y='price',data=df)
df.corr()['bathrooms']['price']
df['bathrooms'].mean()
df.columns
df['sqft_living']
df['sqft_living'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['sqft_living'],bins=50)
df[df['sqft_living']>6000]
df[df['sqft_living']>6000]['sqft_living'].unique()
sns.scatterplot(x='sqft_living',y='price',data=df)
df.corr()['price']['sqft_living']
sns.scatterplot(x='sqft_living',y='price',data=df)
df.columns
df['sqft_lot']
plt.figure(figsize=(12,6))

sns.distplot(df['sqft_lot'],bins=50,kde=False)
df.corr()['sqft_lot']['price']
sns.scatterplot(x='sqft_lot',y='price',data=df)
df['floors']
df['floors'].nunique()
sns.barplot(x='floors',y='price',data=df)
df.groupby('floors')['price'].mean()
df.groupby('floors')['price'].std()
df['floors'].value_counts()
df.corr()['floors']['price']
df.columns
df['waterfront']
df['waterfront'].nunique()
sns.countplot(x='waterfront',data=df)
sns.barplot(x='waterfront',y='price',data=df)
df[df['waterfront']==1]['waterfront'].count()/len(df)
df.corr()['waterfront']['price']
df.groupby('waterfront')['price'].mean()
df.columns
df['view']
df['view'].nunique()
df['view'].unique()
df['view'].value_counts()
sns.barplot(x='view',y='price',data=df)
df.groupby('view')['price'].mean()
df.groupby('view')['price'].std()
df.corr()['view']['price']
df['condition']
df['condition'].nunique()
df['condition'].unique()
sns.countplot(x='condition',data=df)
df['condition'].value_counts()
sns.barplot(x='condition',y='price',data=df)
df.groupby('condition')['price'].mean()
df.groupby('condition')['price'].std()
df.corr()['condition']['price']
df.groupby('condition')['price'].mean()
df.groupby('condition')['price'].std()
df.columns
df['grade']
df['grade'].nunique()
df['grade'].unique()
plt.figure(figsize=(12,6))

sns.countplot(x='grade',data=df)
df['grade'].value_counts()
plt.figure(figsize=(12,6))

sns.barplot(x='grade',y='price',data=df)
df.groupby('grade')['price'].mean()
df.groupby('grade')['price'].std()
df.groupby('grade')['price'].mean()
df.groupby('grade')['price'].std()
df.corr()['grade']['price']
df.columns
df['sqft_above']
df['sqft_above'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['sqft_above'],kde=False,bins=50)
sns.scatterplot(x='sqft_above',y='price',data=df)
df.corr()['sqft_above']['price']
df['sqft_basement']
df['sqft_basement'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['sqft_basement'],kde=False,bins=50)
sns.scatterplot(x='sqft_basement',y='price',data=df)
df.corr()['sqft_basement']['price']
df.columns
df['yr_built']
df['yr_built'].nunique()
df['yr_built'].value_counts()
plt.figure(figsize=(20,6))

sns.barplot(x='yr_built',y='price',data=df)
sns.scatterplot(x='yr_built',y='price',data=df)
df.corr()['yr_built']['price']
df['yr_renovated']
df['yr_renovated'].nunique()
sns.scatterplot(x='yr_renovated',y='price',data=df)
plt.figure(figsize=(12,6))

sns.distplot(df['yr_renovated'],kde=False,bins=50)
df['yr_renovated'].sort_values().unique()
df['yr_renovated'].value_counts()
df.corr()['yr_renovated']['price']
df.columns
df['zipcode']
df['zipcode'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['zipcode'],kde=False,bins=50)
df['zipcode'].value_counts()
sns.scatterplot(x='zipcode',y='price',data=df)
df.groupby('zipcode')['price'].mean()
df.groupby('zipcode')['price'].std()
df['zipcode'].max()
df['zipcode'].min()
df.corr()['zipcode']['price']
df['lat']
df['lat'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['lat'],kde=False,bins=50)
sns.scatterplot(x='lat',y='price',data=df)
df.corr()['lat']['price']
df['lat'].value_counts()
df['long']
df['long'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['long'],kde=False,bins=50)
df['long'].value_counts()
sns.scatterplot(x='long',y='price',data=df)
df.corr()['long']['price']
df['long'].max()
df['long'].min()
df.columns
df['sqft_living15']
df['sqft_living15'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['sqft_living15'],kde=False,bins=50)
df['sqft_living15'].value_counts()
sns.scatterplot(x='sqft_living15',y='price',data=df)
df.corr()['sqft_living15']['price']
df.groupby('sqft_living15')['price'].mean()
df.groupby('sqft_living15')['price'].mean()
df['sqft_lot15']
df['sqft_lot15'].nunique()
plt.figure(figsize=(12,6))

sns.distplot(df['sqft_lot15'],kde=False,bins=50)
df['sqft_lot15'].value_counts()
df['sqft_lot15'].max()
df['sqft_lot15'].min()
df.corr()['sqft_lot15']['price']
sns.scatterplot(x='sqft_lot15',y='price',data=df)
df.corr()['sqft_lot15']['price']
sns.scatterplot(x='sqft_lot15',y='price',data=df)
df.columns
df['year']
df['year'].nunique()
df['year'].unique()
sns.countplot(x='year',data=df)
df['year'].value_counts()
sns.barplot(x='year',y='price',data=df)
df.groupby('year')['price'].mean()
df.groupby('year')['price'].std()
df.corr()['year']['price']
df['month']
sns.countplot(x='month',data=df)
df['month'].value_counts()
plt.figure(figsize=(12,6))

sns.barplot(x='month',y='price',data=df)
df.groupby('month')['price'].mean()
df.groupby('month')['price'].std()
df.corr()['month']['price']
df['day']
plt.figure(figsize=(12,6))

sns.countplot(x='day',data=df)
df['day'].value_counts()
plt.figure(figsize=(12,6))

sns.barplot(x='day',y='price',data=df)
df.groupby('day')['price'].mean()
df.groupby('day')['price'].std()
df.corr()['day']['price']
df.head()
plt.figure(figsize=(12,6))

sns.distplot(df['price'],kde=False,bins=50)
df[df['price']>2000000]['price'].count()
len(df)
df = df[df['price']<=2000000]
from sklearn.model_selection import train_test_split
X = df.drop('price',axis=1)

y = df['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
from xgboost import XGBRegressor
xbgr = XGBRegressor()
xbgr.fit(X_train,y_train)
predictions = xbgr.predict(X_test)
print('MAE: ',mean_absolute_error(y_test,predictions))

print('MSE: ',mean_squared_error(y_test,predictions))

print('RMSE: ',mean_squared_error(y_test,predictions)**0.5)

print('EVS: ',explained_variance_score(y_test,predictions))
df['price'].mean()
plt.scatter(y_test,predictions)

plt.plot(y_test,y_test,'r')