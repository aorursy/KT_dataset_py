# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt



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
data.dtypes
data.describe()[['Sales','Customers']].loc['mean']
print(data.describe()[['Sales','Customers']].loc['min'])

data.describe()[['Sales','Customers']].loc['max']
print(data.Store.nunique())

data.Store.value_counts().head(50).plot.bar() #top 50 stores have 942 records
data.Store.value_counts().tail(50).plot.bar() # bottom 50 only have 758 records
data.DayOfWeek.value_counts()
data.Open.value_counts()
data.Promo.value_counts()
data.StateHoliday.value_counts()
data.SchoolHoliday.value_counts()
store.isna().sum() #there are so many missing values
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')

store_id=data.Store.unique()[0] #6th store. we change the position to see each store data

print(store_id)

store_rows=data[data['Store']==store_id] #copying the data whose store ID is 6

print(store_rows.shape)

store_rows.resample('1D',on='Date')['Sales'].sum().plot.line(figsize=(14,4))
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

store_test_rows=test[test['Store']==store_id]

store_test_rows['Date'].min(),store_test_rows['Date'].max()
store_test_rows['Open'].value_counts()
store_rows['Sales'].plot.hist()
store[store['Store']==store_id].T
store[~store['Promo2SinceYear'].isna()].iloc[0] 
store.isna().sum()
store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)
store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])

store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())
store.isna().sum()
store['Promo2SinceYear'].mode()
data_merged=data.merge(store,on='Store',how='left')
print(data.shape)

print(data_merged.shape)

print(data_merged.isna().sum().sum()) #cross checking if there are any missing values
data_merged
data_merged.dtypes # 3 categorical column, 1 date column, rest all numerical
data_merged['day']=data_merged['Date'].dt.day

data_merged['month']=data_merged['Date'].dt.month

data_merged['year']=data_merged['Date'].dt.year

#data_merged['Date'].dt.strftime('%a') - This is already in the data
data_merged.dtypes

#StateHoliday, StoreType,Assortment,PromoInterval
data_merged['StateHoliday'].unique()

data_merged['StateHoliday']=data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

data_merged['StateHoliday']=data_merged['StateHoliday'].astype(int)
data_merged.dtypes
data_merged['Assortment'].unique()

data_merged['Assortment']=data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment']=data_merged['Assortment'].astype(int)
data_merged.PromoInterval.unique()
map_promo={'Jan,Apr,Jul,Oct':1, 'Feb,May,Aug,Nov':2, 'Mar,Jun,Sept,Dec':3}

data_merged['PromoInterval']=data_merged['PromoInterval'].map(map_promo)

data_merged
# Train and validate Split

features= data_merged.columns.drop(['Sales','Date','Customers'])

from sklearn.model_selection import train_test_split

train_x,validate_x,train_y,validate_y = train_test_split(data_merged[features],np.log(data_merged['Sales']+1),test_size=0.2,random_state=1)

train_x.shape,validate_x.shape,train_y.shape,validate_y.shape
from sklearn.tree import DecisionTreeRegressor

model_dt=DecisionTreeRegressor(max_depth=11,random_state=1).fit(train_x,train_y)

validate_y_pred=model_dt.predict(validate_x)
def draw_tree(model, columns):

    import pydotplus

    from sklearn.externals.six import StringIO

    from IPython.display import Image

    import os

    from sklearn import tree

    

    graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'

    os.environ["PATH"] += os.pathsep + graphviz_path



    dot_data = StringIO()

    tree.export_graphviz(model,

                         out_file=dot_data,

                         feature_names=columns)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

    return Image(graph.create_png())
!pip install pydotplus
draw_tree(model_dt,features)
pd.Series(np.log(data_merged['Sales']+1)).plot.hist()
validate_y_pred = model_dt.predict(validate_x)

from sklearn.metrics import mean_squared_error

validate_y_inv=np.exp(validate_y)-1 #becaused we added +1 while log transformation

validate_y_pred_inv=np.exp(validate_y_pred)-1

np.sqrt(mean_squared_error(validate_y_inv,validate_y_pred_inv))
model_dt.feature_importances_
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

yvalues=model_dt.feature_importances_

xvalues=features

plt.bar(yvalues,xvalues)
data_merged.corr().loc['Sales'].sort_values(ascending=False)
plt.figure(figsize=(10,5))

#plt.bar(features,model_dt.feature_importances_)

pd.Series(model_dt.feature_importances_,index=features)
#Hyperparameter tuning



# from sklearn.model_selection import GridSearchCV



# parameters={'max_depth':list(range(5,20))}

# base_model=DecisionTreeRegressor()

# cv_model=GridSearchCV(base_model,param_grid=parameters,cv=5,return_train_score=True).fit(train_x,train_y)

# parameters
# cv_model.best_params_ 

# #few times, this will overfit. so instead of directly going with best params, 

# #plot the graph and decide optimal parameters
# pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)#[['param_max_depth','mean_test_score']]

# #differnt types tried with different max depth
# df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)

# df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

# df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

stores_avg_custs=data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)

test_1=test.merge(stores_avg_custs,on='Store',how='left')

test.shape,test_1.shape
test_merged=test_1.merge(store,on='Store',how='left')

test_merged.isna().sum()
test_merged['Open']=test_merged['Open'].fillna(1)

test_merged['Date']=pd.to_datetime(test_merged['Date'],format='%Y-%m-%d')

test_merged['day']=test_merged['Date'].dt.day

test_merged['month']=test_merged['Date'].dt.month

test_merged['year']=test_merged['Date'].dt.year

test_merged['StateHoliday']=test_merged['StateHoliday'].map({'0':0,'a':1})

test_merged['StateHoliday']=test_merged['StateHoliday'].astype(int)

test_merged['Assortment']=test_merged['Assortment'].map({'a':1,'b':2,'c':3})

test_merged['Assortment']=test_merged['Assortment'].astype(int)

test_merged['StoreType']=test_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

test_merged['StoreType']=test_merged['StoreType'].astype(int)

map_promo = {'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

test_merged['PromoInterval']=test_merged['PromoInterval'].map(map_promo)
test_pred=model_dt.predict(test_merged[features])

test_pred_inv=np.exp(test_pred)-1
submission=pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')
submission.head()
submission_predicted=pd.DataFrame({'Id': test['Id'],'Sales':test_pred_inv})
submission_predicted.head()
submission_predicted.to_csv('submission.csv',index=False)
# Credit: kaggle.com

def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe





validate_y_inv=np.exp(validate_y)-1

validate_y_pred_inv=np.exp(validate_y_pred)-1

rmse_val=np.sqrt(mean_squared_error(validate_y_inv,validate_y_pred_inv))

rmspe_val=rmspe(validate_y_inv,validate_y_pred_inv)

print(rmse_val,rmspe_val)