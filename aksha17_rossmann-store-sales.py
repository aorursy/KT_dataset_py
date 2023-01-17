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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

store= pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

test= pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
print(data.shape)

print(store.shape)
data.head()
store.head()
data.info()

# data.dtypes
data.describe(include='object')
data.describe()[['Sales','Customers']]
data.describe()[['Sales','Customers']].loc['mean']
data.describe()[['Sales','Customers']].loc['min']
data.describe()[['Sales','Customers']].loc['max']
data.Store.nunique()
data.head()

data.Store.value_counts().head(50).plot.bar()
data.Store.value_counts().tail(50).plot.bar()
data.Store.value_counts()
data.DayOfWeek.value_counts()
data.Open.value_counts()
data.Promo.value_counts()
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')

store_id= data.Store.unique()[0]

print(store_id)

store_rows=data[data['Store']==store_id]

print(store_rows.shape)

# store_rows.resample('1D',on='Date')['Sales'].sum().plot.line(figsize=(14,4))
# store_rows[store_rows['Sales']==0]
test['Date']=pd.to_datetime(test['Date'],format='%Y-%m-%d')

store_test_rows = test[test['Store']==store_id]

store_test_rows['Date'].min(),store_test_rows['Date'].max()
store_rows['Sales'].plot.hist()

# it is slightly skewed.
data['Sales'].plot.hist()

# it is slightly skewed.
store.head()
# store.isna.sum()
store_id=store[store['Store']==1].T
store[~store['Promo2SinceYear'].isna()].iloc[0]
# Method1

store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

store['Promo2SinceWeek']= store['Promo2SinceWeek'].fillna(0)

store['Promo2SinceYear']= store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

store['PromoInterval']= store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0]) 



store['CompetitionDistance']=store['CompetitionDistance'].fillna(store['CompetitionDistance'].max())

store['CompetitionOpenSinceMonth']= store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']= store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])

store.isna().sum()

      
data_merged = data.merge(store, on='Store',how='left')

print(data.shape)

print(data_merged.shape)

print(data_merged.isna().sum().sum()) #to cross check if there are any missing values
# encoding

# 3 categorical column,1 date column, rest are numerical

# data_merged.dtypes

data_merged['day']=data_merged['Date'].dt.day

data_merged['month']=data_merged['Date'].dt.month

data_merged['year']=data_merged['Date'].dt.year

#data_merged['dayofweek']=data_merged['Date'].dt.strftime('%a')

# Decision tress - label encoding should be used.

# regression - one hot encoding must be used.
# data_merged.dtypes

# StateHoliday,StoreType,Assortment,PromoInterval

data_merged['StateHoliday'].unique()

# for creating dummy variables - label encoding is used

data_merged['StateHoliday']=data_merged['StateHoliday'].map({'0':0,0:0,'a':1,'b':2,'c':3})

data_merged['StateHoliday']=data_merged['StateHoliday'].astype(int)

data_merged
# encoding assorted

data_merged['Assortment']

# for creating dummy variables - label encoding is used

data_merged['Assortment']=data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment']=data_merged['Assortment'].astype(int)

data_merged
data_merged['StoreType'].unique()

data_merged['StoreType']=data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

data_merged['StoreType']=data_merged['StoreType'].astype(int)

data_merged
data_merged['PromoInterval'].unique()

map_promo = {'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

data_merged['PromoInterval']=data_merged['PromoInterval'].map(map_promo)

data_merged
# Train and validate Split

features= data_merged.columns.drop(['Sales','Customers','Date'])

from sklearn.model_selection import train_test_split

train_x,validate_x,train_y,validate_y = train_test_split(data_merged[features],np.log(data_merged['Sales']+1),test_size=0.2,random_state=1)

train_x.shape,validate_x.shape,train_y.shape,validate_y.shape
# from sklearn.tree import DecisionTreeRegressor



# model_dt = DecisionTreeRegressor(max_depth=20,random_state=1).fit(train_x,train_y)

# validate_y_pred = model_dt.predict(validate_x)



from sklearn.tree import DecisionTreeRegressor

model_dt=DecisionTreeRegressor(max_depth=11,random_state=1).fit(train_x,train_y)

validate_y_pred=model_dt.predict(validate_x)
# !pip install pydotplus
# def draw_tree(model, columns):

#     import pydotplus

#     from sklearn.externals.six import StringIO

#     from IPython.display import Image

#     import os

#     from sklearn import tree

    

#     graphviz_path = 'C:\Program Files (x86)\Graphviz2.38/bin/'

#     os.environ["PATH"] += os.pathsep + graphviz_path



#     dot_data = StringIO()

#     tree.export_graphviz(model,

#                          out_file=dot_data,

#                          feature_names=columns)

#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

#     return Image(graph.create_png())
#draw_tree(model_dt,features)
validate_y_pred = model_dt.predict(validate_x)

from sklearn.metrics import mean_squared_error

validate_y_inv = np.exp(validate_y) - 1

validate_y_pred_inv = np.exp(validate_y_pred) - 1

rmse_val=np.sqrt(mean_squared_error(validate_y_inv , validate_y_pred_inv))



def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w



def rmspe(y, yhat):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe



validate_y_inv = np.exp(validate_y) - 1

validate_y_pred_inv = np.exp(validate_y_pred) - 1

rmse_val=np.sqrt(mean_squared_error(validate_y_inv , validate_y_pred_inv))

rmspe_val=rmspe(validate_y_inv , validate_y_pred_inv)

print(rmse_val,rmspe_val)
# submitting the train on test data set
model_dt.feature_importances_
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.barh(features,model_dt.feature_importances_)

pd.Series(model_dt.feature_importances_,index=features)
#Hyperparameter tuning

from sklearn.model_selection import GridSearchCV



parameters={'max_depth':list(range(5,20))}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=parameters,cv=5,return_train_score=True).fit(train_x,train_y)

parameters
cv_model.best_params_
stores_avg_cust = data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)

test_1 = test.merge(stores_avg_cust,on='Store',how='left')

test.shape,test_1.shape

test_merged = test_1.merge(store,on='Store',how='inner')

test_merged['Open']=test_merged['Open'].fillna(1)

test_merged['Date']=pd.to_datetime(test_merged["Date"],format='%Y-%m-%d')

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

df_cv_results=pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)[['param_max_depth','mean_test_score','mean_train_score']]

import matplotlib.pyplot as plt

df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

plt.legend(['Test Scores','Train Scores'])
df_cv_results[df_cv_results['param_max_depth']==11].T
test_merged
test_pred = model_dt.predict(test_merged[features])

test_pred_inv = np.exp(test_pred) - 1
submission = pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')

submission_predicted = pd.DataFrame({'Id':test['Id'],'Sales':test_pred_inv})

submission_predicted.to_csv('submission.csv',index=False)

submission_predicted.head()
from sklearn.model_selection import GridSearchCV

def get_rmspe_score(model,input_values,y_actual):

    y_predicted=model.predict(input_values)

    y_actual=np.exp(y_actual)-1

    y_predicted=np.exp(y_predicted)-1

    score=rmspe(y_actual,y_predicted)

    return score



parameters={'max_depth':list(range(5,8))}

base_model=DecisionTreeRegressor()

cv_model=GridSearchCV(base_model,param_grid=parameters,cv=5,return_train_score=True,scoring=get_rmspe_score).fit(train_x,train_y)

pd.DataFrame(cv_model.cv_results_)[['params','mean_test_score','mean_train_score']]