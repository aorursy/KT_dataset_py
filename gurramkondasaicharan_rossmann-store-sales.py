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
data=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

test=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

print(data.shape)

print(store.shape)

print(test.shape)
data.head()
store.head()
test.head()
data.info()
data.describe()[['Sales','Customers']]
data.Store.nunique()

# data.Store.value_counts().tail(50).plot.bar()

# data.Store.value_counts().head(50).plot.bar()

data.Store.value_counts()
data.Promo.value_counts()
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')

store_id=data.Store.unique()[0]

print(store_id)

store_rows=data[data['Store']==store_id]

print(store_rows.shape)
store_rows.resample('1D', on='Date')['Sales'].sum().plot.line(figsize=(14,4))
store_rows[store_rows['Sales']==0]
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

store_test_rows=test[test['Store']==store_id]

store_test_rows['Date'].min(),store_test_rows['Date'].max()
store_rows['Sales'].plot.hist()
store.head()

store[store['Store']==store_id].T
store[~store['Promo2SinceYear'].isna()].iloc[0]
store.isna().sum()
#Method-1(Filling missing valiues(Data Imputation))

store=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')



store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)

store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])



store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])
store.isna().sum()
data_merged=data.merge(store,on='Store',how='left')

print(data.shape)

print(data_merged.shape)
# Encoding

# 4-Categorical Column, 1 Date column, rest are numerical



data_merged.dtypes
data_merged['day']=data_merged['Date'].dt.day

data_merged['month']=data_merged['Date'].dt.month

data_merged['year']=data_merged['Date'].dt.year

#data_merged['dayofweek']=data_merged['Date'].dt.strftime('%a') #This is already given in the data just for my knowledge
# StateHoliday,  Storetype, Assortment, PromoInterval are the categorical columns



data_merged['StateHoliday'].unique()
#Label Encoding to StateHoliday



data_merged['StateHoliday']=data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

data_merged['StateHoliday']=data_merged['StateHoliday'].astype(int)

data_merged['StateHoliday'].dtypes
data_merged['Assortment'].unique()

data_merged['Assortment']=data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment']=data_merged['Assortment'].astype(int)
data_merged['StoreType'].unique()

data_merged['StoreType']=data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

data_merged['StoreType']=data_merged['StoreType'].astype(int)
data_merged['PromoInterval'].unique()

data_merged['PromoInterval']=data_merged['PromoInterval'].map({'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3})

data_merged['PromoInterval']=data_merged['PromoInterval'].astype(int)
## Train & Validation Split



x=data_merged.drop(['Sales','Date'],axis=1)

y=np.log(data_merged['Sales']+1)

from sklearn.model_selection import train_test_split

train_x,validate_x,train_y,validate_y=train_test_split(x,y,test_size=0.2,random_state=0)

train_x.shape,validate_x.shape,train_y.shape,validate_y.shape
"""

#Hyperparameters Tunning(Gridsearchcv)



from sklearn.model_selection import GridSearchCV



param={'max_depth':range(5,20)}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=param,cv=5,return_train_score=True).fit(train_x,train_y)

"""
# cv_model.best_params_
"""

df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score','mean_train_score']]

df_cv_results

"""
"""

df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

plt.legend()

plt.grid()

"""
from sklearn.tree import DecisionTreeRegressor



model_dt=DecisionTreeRegressor(max_depth=11,random_state=0)

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
y_pred=model_dt.predict(validate_x)



validate_y_inv=np.exp(validate_y)-1

y_pred_inv=np.exp(y_pred)-1

from sklearn.metrics import mean_squared_error , r2_score

print('R-Squared',r2_score(validate_y_inv,y_pred_inv))

print('RMSE',np.sqrt(mean_squared_error(validate_y_inv,y_pred_inv)))

print('RMSPE',rmspe(validate_y_inv,y_pred_inv))
test.head()
model_dt.feature_importances_

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.bar(x.columns,model_dt.feature_importances_)

plt.xticks(rotation=90)
stores_avg_cust=data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)

test_1=test.merge(stores_avg_cust,on='Store',how='left')

test.shape,test_1.shape
test_merged=test_1.merge(store,on='Store',how='left')

test_merged['Open']=test_merged['Open'].fillna(1)

test_merged['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

test_merged['day']=test_merged['Date'].dt.day

test_merged['month']=test_merged['Date'].dt.month

test_merged['year']=test_merged['Date'].dt.year

test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)

test_merged['Assortment']=test_merged['Assortment'].map({'a':1,'b':2,'c':3})

test_merged['Assortment']=test_merged['Assortment'].astype(int)

test_merged['StoreType']=test_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

test_merged['StoreType']=test_merged['StoreType'].astype(int)

test_merged['PromoInterval']=test_merged['PromoInterval'].map({'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3})

test_merged['PromoInterval']=test_merged['PromoInterval'].astype(int)
test_merged.info()
test_pred=model_dt.predict(test_merged[x.columns])

test_pred_inv=np.exp(test_pred)-1
submission_predicted=pd.DataFrame({'Id':test['Id'],'Sales':test_pred_inv})

submission_predicted.head()
submission_predicted.to_csv('submission.csv',index = False)
## Train & Validation Split ( we are deleting customers feature from the data because of the model over fit when we test in test data)

"""

x=data_merged.drop(['Sales','Customers','Date'],axis=1)

y=np.log(data_merged['Sales']+1)

from sklearn.model_selection import train_test_split

train_x,validate_x,train_y,validate_y=train_test_split(x,y,test_size=0.2,random_state=0)

train_x.shape,validate_x.shape,train_y.shape,validate_y.shape

"""
#Hyperparameters Tunning(Gridsearchcv)

"""

from sklearn.model_selection import GridSearchCV



param={'max_depth':range(5,20)}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=param,cv=5,return_train_score=True).fit(train_x,train_y)

"""
# cv_model.best_params_
"""

df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score','mean_train_score']]

df_cv_results

"""
"""

df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

plt.legend()

plt.grid()

"""
"""

model_dt=DecisionTreeRegressor(max_depth=11,random_state=0)

model_dt.fit(train_x,train_y)

"""
"""

y_pred=model_dt.predict(validate_x)



validate_y_inv=np.exp(validate_y)-1

y_pred_inv=np.exp(y_pred)-1

from sklearn.metrics import mean_squared_error , r2_score

print('R-Squared',r2_score(validate_y_inv,y_pred_inv))

print('RMSE',np.sqrt(mean_squared_error(validate_y_inv,y_pred_inv)))

print('RMSPE',rmspe(validate_y_inv,y_pred_inv))

"""