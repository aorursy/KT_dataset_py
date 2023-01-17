import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

print(data.shape)

print(store.shape)

print(test.shape)
store.head()
data.head()
test.head()
data.info()
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')

#store_id=data.Store.unique()[0]

#print(store_id)

store_rows=data[data['Store']==1]

print(store_rows.shape)

store_rows.resample('1D',on='Date')['Sales'].sum().plot.line(figsize=(14,4))
#CHECKING THE STARTING AND END DATE

print(data['Date'].min())

print(data['Date'].max())
data['Sales'].plot.hist()    #right skewed
 # CHECKING MISSING VALUES

store.isna().sum()
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')



store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)

store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

store['PromoInterval'] = store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])



store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])
store['Promo2SinceYear'].mode().iloc[0]
data_merged = data.merge(store,on='Store',how='left')

data_merged.head()
#partitioning the date into day,month,year

data_merged['day'] = data_merged['Date'].dt.day

data_merged['month'] = data_merged['Date'].dt.month

data_merged['year'] = data_merged['Date'].dt.year

#data_merged['Date'].dt.strftime('%a')-
#CATEGORICAL-stateholiday,storetype,assortment,promointerval

data_merged['StateHoliday'] = data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

data_merged['StateHoliday'] = data_merged['StateHoliday'].astype(int)
data_merged['Assortment'] = data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment'] = data_merged['Assortment'].astype(int)
data_merged['StoreType'] = data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

data_merged['StoreType'] = data_merged['StoreType'].astype(int)
map_promo={'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

data_merged['PromoInterval']=data_merged['PromoInterval'].map(map_promo)

data_merged['PromoInterval'] = data_merged['PromoInterval'].astype(int)
# Train & Test split

from sklearn.model_selection import train_test_split

X=data_merged.drop(['Date','Sales'],axis=1)

y=np.log(data_merged['Sales']+1)

train_x,validate_x,train_y,validate_y = train_test_split(X,y,test_size=0.20,random_state=1)

train_x.shape,validate_x.shape,train_y.shape,validate_y.shape
from sklearn.tree import DecisionTreeRegressor



model_dt=DecisionTreeRegressor(max_depth=10,random_state=1)

model_dt.fit(train_x,train_y)
#Code for RMSPE Value

def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe
from sklearn.metrics import r2_score, mean_squared_error

y_pred=model_dt.predict(validate_x)

y_pred_exp=np.exp(y_pred)-1

validate_y_exp=np.exp(validate_y)-1

print("R-squared:", r2_score(validate_y_exp, y_pred_exp))

print("RMSE:", np.sqrt(mean_squared_error(validate_y_exp, y_pred_exp)))

print('RMSPE',rmspe(validate_y_exp, y_pred_exp))

#checking the feature which has most importance

import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))

plt.bar(X.columns,model_dt.feature_importances_)
stores_avg_cust=data.groupby(['Store'])['Customers'].mean().reset_index().astype(int)

stores_avg_cust

test_1=test.merge(stores_avg_cust,on='Store',how='left')

test_merg=test_1.merge(store,on='Store',how='left')

test_merg['Open']=test_merg['Open'].fillna(1)

test_merg['Date']=pd.to_datetime(test_merg['Date'],format='%Y-%m-%d')

test_merg['day'] = test_merg['Date'].dt.day

test_merg['month'] = test_merg['Date'].dt.month

test_merg['year'] = test_merg['Date'].dt.year
test_merg.describe()
#CATEGORICAL-stateholiday,storetype,assortment,promointerval

test_merg['StateHoliday'] = test_merg['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

test_merg['StateHoliday'] = test_merg['StateHoliday'].astype(int)



test_merg['Assortment'] = test_merg['Assortment'].map({'a':1,'b':2,'c':3})

test_merg['Assortment'] = test_merg['Assortment'].astype(int)



test_merg['StoreType'] = test_merg['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

test_merg['StoreType'] = test_merg['StoreType'].astype(int)



map_promo={'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

test_merg['PromoInterval']=test_merg['PromoInterval'].map(map_promo)

test_merg['PromoInterval'] = test_merg['PromoInterval'].astype(int)
test_merged=test_merg.drop(['Id','Date'],axis=1)

test_merged.describe()
test_pred=model_dt.predict(test_merg[X.columns])

test_pred_inv=np.exp(test_pred)-1

submission_predict = pd.DataFrame({'ID':test['Id'],'Sales':test_pred_inv})
submission_predict
submission_predict.to_csv('submission.csv',index=False)