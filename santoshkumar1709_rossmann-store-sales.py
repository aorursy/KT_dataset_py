# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")

store=pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")

test=pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv")

submission=pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv")

print(submission.shape)

print(test.shape)

print(data.shape)

print(store.shape)
data.head()
test.head()
submission.head()
data.shape
data.dtypes
data.describe(include='object').T
data.describe()[['Sales','Customers']].loc['max']
data.Store.nunique()
data.Store.value_counts().head(100).plot.bar()

data.Store.value_counts().tail(100).plot.bar()
data.DayOfWeek.value_counts()
data.Open.value_counts()
import matplotlib.pyplot as plt

data.Promo.value_counts().plot(kind='bar')

data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')

store_id=data.Store.unique()[0]

print(store_id)

store_rows=data[data['Store']==store_id]

print(store_rows.shape)

store_rows[store_rows['Sales']==0]
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

store_test_rows=test[test['Store']==store_id]

store_test_rows['Date'].min(),store_test_rows['Date'].max()
store_test_rows['Open'].value_counts()
store_rows['Sales'].plot.hist()
data['Sales'].plot.hist()
store.head()

store[store['Store']==store_id].T
store.isna().sum()
store[~store['Promo2SinceYear'].isna()].iloc[0]
# Missing value treatment

store.isna().sum()
#Method 1

store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)

store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])
store['CompetitionDistance']=store['CompetitionDistance'].fillna(0)

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(

    store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])

store.isna().sum()
data_merged=data.merge(store,on='Store',how='left')

print(data.shape)

print(data_merged.shape)
data_merged.isna().sum().sum()
## Encoding



# 3 categorical columns and 1 date column and rest are numerical columns

data_merged.dtypes
data_merged['Day']=data_merged['Date'].dt.day

data_merged['month']=data_merged['Date'].dt.month

data_merged['year']=data_merged['Date'].dt.year

# data_merged['weekday']=data_merged['Date'].dt.strftime('%a')
# data_merged.dtypes

# StateHoliday,StoreType,Assortment,PromoInterval

data_merged['StateHoliday']=data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

data_merged['StateHoliday']=data_merged['StateHoliday'].astype(int)

data_merged['Assortment']=data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment']=data_merged['Assortment'].astype(int)
data_merged['StoreType']=data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

data_merged['StoreType']=data_merged['StoreType'].astype(int)

map_promo={'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

data_merged['PromoInterval']=data_merged['PromoInterval'].map(map_promo)
## Train And Validate Split



features=data_merged.columns.drop(['Sales','Date','Customers'])



from sklearn.model_selection import train_test_split



train_x,validate_x,train_y,validate_y=train_test_split(data_merged[features],np.log(data_merged['Sales']+1),

                                                      test_size=0.2,

                                                      random_state=1)



train_x.shape,validate_x.shape,train_y.shape,validate_y.shape
from sklearn.tree import DecisionTreeRegressor

model_dt=DecisionTreeRegressor(max_depth=10,random_state=1).fit(train_x,train_y)

validate_y_pred=model_dt.predict(validate_x)



from sklearn.metrics import r2_score,mean_squared_error

validate_y_inv=np.exp(validate_y)-1

validate_y_pred_inv=np.exp(validate_y_pred)-1

np.sqrt(mean_squared_error(validate_y_inv,validate_y_pred_inv))



import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))



yvalues=model_dt.feature_importances_

xvalues=features

plt.barh(xvalues,yvalues)

test.head()
stores_avg_custs=data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)

test_1=test.merge(stores_avg_custs,on='Store',how='left')

test.shape,test_1.shape

test_merged=test_1.merge(store,on='Store',how='inner')

test_merged['Open']=test_merged['Open'].fillna(1)

test_merged['Date']=pd.to_datetime(test_merged['Date'],format='%Y-%m-%d')

test_merged['Day']=test_merged['Date'].dt.day

test_merged['month']=test_merged['Date'].dt.month

test_merged['year']=test_merged['Date'].dt.year



# data_merged.dtypes

# StateHoliday,StoreType,Assortment,PromoInterval

test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,'a':1})

test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)



test_merged['Assortment']=test_merged['Assortment'].map({'a':1,'b':2,'c':3})

test_merged['Assortment']=test_merged['Assortment'].astype(int)



test_merged['StoreType']=test_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

test_merged['StoreType']=test_merged['StoreType'].astype(int)



map_promo={'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

test_merged['PromoInterval']=test_merged['PromoInterval'].map(map_promo)





test_pred=model_dt.predict(test_merged[features])

test_pred_inv=np.exp(test_pred)-1

submission_predicted=pd.DataFrame({'Id':test['Id'],'sales':test_pred_inv})

submission_predicted.head()

submission_predicted.to_csv('submission.csv',index=False)

submission_predicted.head()
