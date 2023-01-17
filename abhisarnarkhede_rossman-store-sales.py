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
data = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')

store= pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

test= pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')

submission=pd.read_csv('/kaggle/input/rossmann-store-sales/sample_submission.csv')

print(data.shape)

print(store.shape)

print(test.shape)

print(submission.shape)
data.head()
store.head()
test.head()
submission.head()
data.shape
data.dtypes
data.describe()
data.describe()[['Sales','Customers']]
data.describe()[['Sales','Customers']].loc['mean']
data.describe()[['Sales','Customers']].loc['min']
data.describe()[['Sales','Customers']].loc['max']
data.head()
data.Store.nunique()

data.Store.value_counts().head(50).plot.bar()

data.Store.value_counts().tail(50).plot.bar()
data.Store.value_counts()
len(data.Store)
data.DayOfWeek.value_counts()
data.Open.value_counts()
data.Promo.value_counts()
data.isna().sum()

test.isna().sum()

submission.isna().sum()
data['Date'] = pd.to_datetime(data['Date'], format = '%Y-%m-%d')



store_id = data.Store.unique()[0]

print(store_id)

store_rows = data[data['Store'] == store_id]

print(store_rows.shape)

store_rows.resample('1D', on='Date')['Sales'].sum().plot.line(figsize=(14,4))
store_rows[store_rows['Sales']==0]
test['Date'] = pd.to_datetime(test['Date'], format = '%Y-%m-%d')

store_test_rows = test[test['Store'] == store_id]

store_test_rows['Date'].min(),store_test_rows['Date'].max()
store_test_rows['Open'].value_counts()
store_rows['Sales'].plot.hist()
store[store['Store']==store_id].T
store[~store['Promo2SinceYear'].isna()].iloc[0]
#Method 1

store['Promo2SinceWeek']=store['Promo2SinceWeek'].fillna(0)

store['Promo2SinceYear']=store['Promo2SinceYear'].fillna(store['Promo2SinceYear'].mode().iloc[0])

store['PromoInterval']=store['PromoInterval'].fillna(store['PromoInterval'].mode().iloc[0])

store['CompetitionDistance']=store['CompetitionDistance'].fillna(0)

store['CompetitionOpenSinceMonth']=store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].mode().iloc[0])

store['CompetitionOpenSinceYear']=store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].mode().iloc[0])

store.isna().sum()
store['Promo2SinceYear'].mode()


!pip install pydotplus


# merge the data train and store

data_merged = data.merge(store,on = 'Store',how = 'left')

print(data.shape)

print(data_merged.shape)

print(data_merged.isnull().sum().sum()) # cross check if there are any missing values

data_merged['day'] = data_merged['Date'].dt.day

data_merged['month'] = data_merged['Date'].dt.month

data_merged['year'] = data_merged['Date'].dt.year

#data_merged['weekday'] = data_merged['Date'].dt.strftime(%a)  This is already in data

data_merged['StateHoliday'] = data_merged['StateHoliday'].map({'a':1,'b':2,'c':3,'0':0,0:0})

data_merged['StateHoliday'] = data_merged['StateHoliday'].astype(int)

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

model_dt  = DecisionTreeRegressor(max_depth = 3, random_state = 42).fit(X_train,y_train)

y_pred = model_dt.predict(X_test)

y_inv = np.exp(y_test)-1

y_pred_inv = np.exp(y_pred)-1

np.sqrt(mean_squared_error(y_inv,y_pred_inv))


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

draw_tree(model_dt,X.columns)
data_avg_cust = data.groupby(['Store'])[['Customers']].mean().reset_index().astype(int)

test_1 = test.merge(data_avg_cust,on = 'Store',how = 'left')

test.shape,test_1.shape
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


# submission_predicted = pd.DataFrame({'Id' : test['Id'],'Sales':test_pred_inv })

# submission_predicted.head()
# submission_predicted.to_csv('submission.csv',index = False)
# from sklearn.ensemble import RandomForestRegressor

# import xgboost as xgb

# from sklearn.model_selection import GridSearchCV

# ## Hyperparameter Tuning

# params = {'max_depth': list(range(5,15))}

# base_model = DecisionTreeRegressor()

# cv_model = GridSearchCV(base_model,param_grid = params,cv = 5,return_train_score=True).fit(X_train,y_train)

# import matplotlib.pyplot as plt

# df_cv_results = pd.DataFrame(cv_model.cv_results_).sort_values(by='mean_test_score',ascending=False)

# df_cv_results.set_index('param_max_depth')['mean_test_score'].plot.line()

# df_cv_results.set_index('param_max_depth')['mean_train_score'].plot.line()

# plt.show()
model_dt  = DecisionTreeRegressor(max_depth = 11, random_state = 42).fit(X_train,y_train)

y_pred = model_dt.predict(X_test)
y_inv = np.exp(y_test)-1

y_pred_inv = np.exp(y_pred)-1

np.sqrt(mean_squared_error(y_inv,y_pred_inv))

test_pred = model_dt.predict(test_merged[features])

test_pred_inv = np.exp(test_pred)-1


submission_predicted = pd.DataFrame({'Id' : test['Id'],'Sales':test_pred_inv })

submission_predicted.head()
submission_predicted.to_csv('submission.csv',index = False)