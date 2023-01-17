import datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go



from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor

from sklearn.model_selection import train_test_split



# Importing everything from forecasting quality metrics

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error



import warnings

warnings.filterwarnings("ignore") # ignoring annoying warnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read data

features = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/features.csv.zip')

train = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

test = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv')

sample_submission = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')

features.head()
pd.DataFrame(features.dtypes, columns=['Type'])
pd.DataFrame(train.dtypes, columns=['Type'])
pd.DataFrame(stores.dtypes, columns=['Type'])
# Merge Feature and Store

feat_str = features.merge(stores, how='inner', on='Store')

feat_str.head()
train_to = pd.merge(train, feat_str)

train_to.head()
null_columns = (train_to.isnull().sum(axis = 0)/len(train_to)).sort_values(ascending=False).index

null_data = pd.concat([

    train_to.isnull().sum(axis = 0),

    (train_to.isnull().sum(axis = 0)/len(train_to)).sort_values(ascending=False),

    train_to.loc[:, train_to.columns.isin(list(null_columns))].dtypes], axis=1)

null_data = null_data.rename(columns={0: '# null', 

                                      1: '% null', 

                                      2: 'type'}).sort_values(ascending=False, by = '% null')

null_data = null_data[null_data["# null"]!=0]

null_data
test_to = pd.merge(test, feat_str)

test_to.head()
null_test = (test_to.isnull().sum(axis = 0)/len(test_to)).sort_values(ascending=False).index

null_data_test = pd.concat([

    test_to.isnull().sum(axis = 0),

    (test_to.isnull().sum(axis = 0)/len(test_to)).sort_values(ascending=False),

    test_to.loc[:, test_to.columns.isin(list(null_test))].dtypes], axis=1)

null_data_test = null_data_test.rename(columns={0: '# null', 

                                      1: '% null', 

                                      2: 'type'}).sort_values(ascending=False, by = '% null')

null_data_test = null_data_test[null_data_test["# null"]!=0]

null_data_test
del features, train, stores, test
# Split date into year, month, day, days

train = train_to.copy()

test = test_to.copy()



train['Date'] = pd.to_datetime(train['Date'])

train['Year'] = pd.to_datetime(train['Date']).dt.year

train['Month'] = pd.to_datetime(train['Date']).dt.month

train['Week'] = pd.to_datetime(train['Date']).dt.week

train['Day'] = pd.to_datetime(train['Date']).dt.day

train.replace({'A': 1, 'B': 2,'C':3},inplace=True)



test['Date'] = pd.to_datetime(test['Date'])

test['Year'] = pd.to_datetime(test['Date']).dt.year

test['Month'] = pd.to_datetime(test['Date']).dt.month

test['Week'] = pd.to_datetime(test['Date']).dt.week

test['Day'] = pd.to_datetime(test['Date']).dt.day

test.replace({'A': 1, 'B': 2,'C':3},inplace=True)

train.head()
test.head()
train_heat = train.drop('Date', axis=1)

corr = train_heat.corr()

f, ax = plt.subplots(figsize=(20, 12))

sns.heatmap(corr, annot=True)
# Pre-processing for week data

weekly_sales_2010 = train[train.Year==2010]['Weekly_Sales'].groupby(train['Week']).mean()

weekly_sales_2011 = train[train.Year==2011]['Weekly_Sales'].groupby(train['Week']).mean()

weekly_sales_2012 = train[train.Year==2012]['Weekly_Sales'].groupby(train['Week']).mean()



# Plot

fig = go.Figure(

    [

        go.Scatter(x = weekly_sales_2010.index, y = weekly_sales_2010.values, mode = 'markers+lines', name="2010"),

        go.Scatter(x = weekly_sales_2011.index, y = weekly_sales_2011.values, mode = 'markers+lines', name="2011"),

        go.Scatter(x = weekly_sales_2012.index, y = weekly_sales_2012.values, mode = 'markers+lines', name="2012"),

    ]

)



fig.update_layout(title='Average Weekly Sales - Per Year',

                   plot_bgcolor='rgb(230, 230,230)',

                   showlegend=True, template="plotly_white")



fig.show()
# Error metric

def WMAE(dataset, real, predicted):

    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)

    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
train = train.drop(['CPI','Unemployment','MarkDown1','MarkDown2','MarkDown3', 'MarkDown4','MarkDown5', 'Date', 'Temperature', 'Fuel_Price'],axis=1)

test = test.drop(['CPI','Unemployment','MarkDown1','MarkDown2','MarkDown3', 'MarkDown4','MarkDown5', 'Date', 'Temperature', 'Fuel_Price'],axis=1)

train.head()
test.head()
t1 = datetime.datetime.now()

X = train.loc[:, train.columns != 'Weekly_Sales']

y = train.loc[:, train.columns == 'Weekly_Sales']

RF = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)

RF.fit(X, y)

t2 = datetime.datetime.now()

print(t2-t1)

test['Weekly_Sales'] = RF.predict(test)

test.head()
sample_submission['Weekly_Sales'] = test['Weekly_Sales']

sample_submission.to_csv('submission.csv',index=False)