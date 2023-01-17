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
!pip install pydotplus
store = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv")

train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")

test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv",parse_dates=[3])
train.shape
train.head()
test.shape
test.head()
store.head()
train.info()
train.describe()
train.describe()[['Sales','Customers']].loc['min']
train.describe()[['Sales','Customers']].loc['max']
#  no. of stores

train.Store.nunique()
train['Store'].value_counts().head(50).plot.bar()
train.DayOfWeek.value_counts()
train.Open.value_counts()
train.isna().sum()
test.isnull().sum()
#date wise line plot for sales

train['Date'] = pd.to_datetime(train['Date'],format = '%Y-%m-%d')

store_id = train.Store.unique()[0]

print(store_id)

store_rows = train[train['Store'] == store_id]

print(store_rows.shape)

store_rows.resample('1D',on = 'Date')['Sales'].sum().plot.line(figsize = (18,8))
#missing values on days

store_rows[store_rows['Sales']==0]
# checking the same for test data

test['Date'] = pd.to_datetime(test['Date'],format = '%Y-%m-%d')

store_test_rows = test[test['Store'] == store_id]

print(store_test_rows.shape)

store_test_rows['Date'].min(), store_test_rows['Date'].max()
store_rows['Sales'].plot.hist(figsize = (14,8))
## Store data

store.isnull().sum()
store.head()
store[store['Store']==store_id].T # here store id was 1
# checking the non null values in store data to make sure what we can fill in the missing values

store[~store['Promo2SinceYear'].isna()].iloc[0]
#method 1

store['Promo2SinceWeek'].fillna(0,inplace = True)

store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode()[0],inplace = True)

store['PromoInterval'].fillna(store['PromoInterval'].mode()[0],inplace = True)
store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode()[0],inplace = True)

store['CompetitionDistance'].fillna(store['CompetitionDistance'].max(),inplace = True)

store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode()[0],inplace = True)

store.isnull().sum()
# merge the data train and store

data_merged = train.merge(store,on = 'Store',how = 'left')

print(train.shape)

print(data_merged.shape)

print(data_merged.isnull().sum().sum()) # cross check if there are any missing values
## Encoding

# 3 categorical columns, 1 date column, rest are numerical

data_merged.dtypes
data_merged['day'] = data_merged['Date'].dt.day

data_merged['month'] = data_merged['Date'].dt.month

data_merged['year'] = data_merged['Date'].dt.year

#data_merged['weekday'] = data_merged['Date'].dt.strftime(%a)  This is already in data
# stateHoliday, StoreType, Assortment, PromoInterval

data_merged['StateHoliday'].unique()
data_merged['StateHoliday'] = data_merged['StateHoliday'].map({'a':1,'b':2,'c':3,'0':0,0:0})

data_merged['StateHoliday'] = data_merged['StateHoliday'].astype(int)
pd.set_option('display.max_columns',None)

data_merged.head()
data_merged['Assortment'] = data_merged['Assortment'].map({'a':1,'b':2,'c':3})

data_merged['Assortment'] = data_merged['Assortment'].astype(int)
data_merged['StoreType'] = data_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

data_merged['StoreType'] = data_merged['StoreType'].astype(int)
map_promo = {'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

data_merged['PromoInterval'] = data_merged['PromoInterval'].map(map_promo)
from sklearn.model_selection import train_test_split
features = data_merged.columns.drop(['Sales','Date'])

X = data_merged[features]

y = np.log(data_merged['Sales']+1)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, mean_squared_error
model_dt  = DecisionTreeRegressor(max_depth = 20, random_state = 42).fit(X_train,y_train)

y_pred = model_dt.predict(X_test)
r2_score(y_test,y_pred)
mean_squared_error(y_test,y_pred)
np.sqrt(mean_squared_error(y_test,y_pred))
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
# draw_tree(model_dt,X.columns)
y_inv = np.exp(y_test)-1

y_pred_inv = np.exp(y_pred)-1

np.sqrt(mean_squared_error(y_inv,y_pred_inv))
test.head()
# import matplotlib.pyplot as plt

# plt.figure(figsize = (18,8))

# plt.bar(X,model_dt.feature_importances_)
train_avg_cust = train.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)

test_1 = test.merge(train_avg_cust,on = 'Store',how = 'left')

test.shape,test_1.shape
test_1.head()
test_merged = test_1.merge(store,on = 'Store',how = 'inner')

test_merged['Open'] = test_merged['Open'].fillna(1)

test_merged['Date'] = pd.to_datetime(test_merged['Date'],format = '%Y-%m-%d')

test_merged['day'] = test_merged['Date'].dt.day

test_merged['month'] = test_merged['Date'].dt.month

test_merged['year'] = test_merged['Date'].dt.year

test_merged['StateHoliday'] = test_merged['StateHoliday'].map({'0':0,'a':1})

test_merged['StateHoliday'] = test_merged['StateHoliday'].astype(int)

test_merged['Assortment'] = test_merged['Assortment'].map({'a':1,'b':2,'c':3})

test_merged['Assortment'] = test_merged['Assortment'].astype(int)

test_merged['StoreType'] = test_merged['StoreType'].map({'a':1,'b':2,'c':3,'d':4})

test_merged['StoreType'] = test_merged['StoreType'].astype(int)

map_promo = {'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}

test_merged['PromoInterval'] = test_merged['PromoInterval'].map(map_promo)



test_pred = model_dt.predict(test_merged[features])

test_pred_inv = np.exp(test_pred)-1

test_merged.head()
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
# rf = RandomForestRegressor(n_jobs = -1)

# param_grid = { 

#         "n_estimators"      : [10,50,100],

#         "max_features"      : ["auto", "sqrt", "log2"],

#         "min_samples_split" : [2,4],

#         "bootstrap": [True, False],

#         "max_depth" : [5,10,20]

#         }



# grid = GridSearchCV(estimator = rf,param_grid = param_grid, cv=3)



# grid.fit(X_train, y_train)



# grid.best_score_ , grid.best_params_

rf = RandomForestRegressor()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
y_inv = np.exp(y_test)-1

y_pred_inv = np.exp(y_pred)-1

np.sqrt(mean_squared_error(y_inv,y_pred_inv))
test_pred = rf.predict(test_merged[features])

test_pred_inv = np.exp(test_pred)-1

submission_predicted = pd.DataFrame({'Id' : test['Id'],'Sales':test_pred_inv })

submission_predicted.head()
submission_predicted.to_csv('submission.csv',index = False)