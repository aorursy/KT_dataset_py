import pandas as pd

from pandas import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df = pd.read_csv('../input/rossmann-store-sales/train.csv', parse_dates = ['Date'], low_memory = False)

df.head()
df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')
df['Hour'] = df['Date'].dt.hour

df['Day_of_Month'] = df['Date'].dt.day

df['Day_of_Week'] = df['Date'].dt.dayofweek

df['Month'] = df['Date'].dt.month
print(df['Date'].min())

print(df['Date'].max())
test = pd.read_csv('../input/rossmann-store-sales/test.csv', parse_dates = True, low_memory = False)

test.head()
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')
test['Hour'] = test['Date'].dt.hour

test['Day_of_Month'] = test['Date'].dt.day

test['Day_of_Week'] = test['Date'].dt.dayofweek

test['Month'] = test['Date'].dt.month
print(test['Date'].min())

print(test['Date'].max())
sns.pointplot(x='Month', y='Sales', data=df)
sns.pointplot(x='Day_of_Week', y='Sales', data=df)
sns.countplot(x = 'Day_of_Week', hue = 'Open', data = df)

plt.title('Store Daily Open Countplot')
sns.pointplot(x='Day_of_Month', y='Sales', data=df)
df['SalesPerCustomer'] = df['Sales']/df['Customers']

df['SalesPerCustomer'].describe()
df.Open.value_counts()
np.sum([df['Sales'] == 0])
#drop closed stores and stores with zero sales

df = df[(df["Open"] != 0) & (df['Sales'] != 0)]
store = pd.read_csv('../input/rossmann-store-sales/store.csv')

store.head(30)
store.isnull().sum()
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())

store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0]) #try 0

store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0]) #try 0

store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0) #try 0

store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0]) #try 0

store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0]) #try 0

store.head()
df_store = pd.merge(df, store, how = 'left', on = 'Store')

df_store.head()
df_store.groupby('StoreType')['Sales'].describe()
df_store.groupby('StoreType')['Customers', 'Sales'].sum()
#sales trends

sns.catplot(data = df_store, x = 'Month', y = "Sales", 

               col = 'StoreType', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'Promo', # per promo in the store in rows

               color = 'c') 
#customer trends

sns.catplot(data = df_store, x = 'Month', y = "Customers", 

               col = 'StoreType', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'Promo', # per promo in the store in rows

               color = 'c')
#sales per customer

sns.catplot(data = df_store, x = 'Month', y = "SalesPerCustomer", 

               col = 'StoreType', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'Promo', # per promo in the store in rows

               color = 'c')
sns.catplot(data = df_store, x = 'Month', y = "Sales", 

               col = 'DayOfWeek', # per store type in cols

               palette = 'plasma',

               hue = 'StoreType',

               row = 'StoreType', # per store type in rows

               color = 'c') 
#stores open on sunday

df_store[(df_store.Open == 1) & (df_store.DayOfWeek == 7)]['Store'].unique()
sns.catplot(data = df_store, x = 'DayOfWeek', y = "Sales", 

               col = 'Promo', 

               row = 'Promo2',

               hue = 'Promo2',

               palette = 'RdPu') 
df_store['StateHoliday'] = df_store['StateHoliday'].map({'0':0 , 0:0 , 'a':1 , 'b':2 , 'c':3})

df_store['StateHoliday'] = df_store['StateHoliday'].astype(int)
df_store['StoreType'] = df_store['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})

df_store['StoreType'] = df_store['StoreType'].astype(int)
df_store.isnull().sum()
df_store['Assortment'] = df_store['Assortment'].map({'a':1 , 'b':2 , 'c':3})

df_store['Assortment'] = df_store['Assortment'].astype(int)
df_store['PromoInterval'] = df_store['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})

df_store['PromoInterval'] = df_store['PromoInterval'].astype(int)
df_store.to_csv('df_merged.csv', index=False)
df_store.isnull().sum()
len(df_store)
test = pd.merge(test, store, how = 'left', on = 'Store')

test.head()
test.isnull().sum()
test.fillna(method='ffill', inplace=True)
test['StateHoliday'] = test['StateHoliday'].map({'0':0 , 0:0 , 'a':1 , 'b':2 , 'c':3})

test['StateHoliday'] = test['StateHoliday'].astype(int)

test['StoreType'] = test['StoreType'].map({'a':1 , 'b':2 , 'c':3 , 'd':4})

test['StoreType'] = test['StoreType'].astype(int)

test['Assortment'] = test['Assortment'].map({'a':1 , 'b':2 , 'c':3})

test['Assortment'] = test['Assortment'].astype(int)

test['PromoInterval'] = test['PromoInterval'].map({'Jan,Apr,Jul,Oct':1 , 'Feb,May,Aug,Nov':2 , 'Mar,Jun,Sept,Dec':3})

test['PromoInterval'] = test['PromoInterval'].astype(int)
test.to_csv('test_merged.csv', index=False)
test = test.drop(['Id','Date'],axis=1)
test.head()
X = df_store.drop(['Date','Sales','Customers', 'SalesPerCustomer'],1)

#Transform Target Variable

y = np.log1p(df_store['Sales'])



from sklearn.model_selection import train_test_split

X_train , X_val , y_train , y_val = train_test_split(X, y , test_size=0.30 , random_state = 1 )
X_train.shape, X_val.shape, y_train.shape, y_val.shape
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=10, n_estimators=200, random_state=42)

gbrt.fit(X_train, y_train)

print(gbrt.score(X_train, y_train))
y_pred = gbrt.predict(X_val)
from sklearn.metrics import r2_score, mean_squared_error

print(r2_score(y_val , y_pred))

print(np.sqrt(mean_squared_error(y_val , y_pred)))
df1 = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})

df1.head(25)
test_pred=gbrt.predict(test[X.columns])

test_pred_inv=np.exp(test_pred)-1
test_pred_inv
#make submission df

prediction = pd.DataFrame(test_pred_inv)

submission = pd.read_csv('../input/rossmann-store-sales/sample_submission.csv')

prediction_df = pd.concat([submission['Id'], prediction], axis=1)

prediction_df.columns=['Id','Sales']

prediction_df.to_csv('Sample_Submission.csv', index=False)
prediction_df.head()