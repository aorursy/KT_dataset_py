import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


import matplotlib.pylab as plt
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
%matplotlib inline
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
train_df = load_df()
test_df = load_df("../input/test.csv")
train_df.shape,test_df.shape
train_df.head()
intersection = list(set(train_df.columns) & set(test_df.columns))
li = list(set(train_df.columns) -set(intersection))

print ('feature not in test but in train:' ,li)

train_df.drop('trafficSource.campaignCode' , axis =1,inplace = True)
train_df.shape,test_df.shape
def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

missing_data_df = check_missing(train_df)
missing_data_test = check_missing(test_df)

print('Missing data in train set: \n' , missing_data_df.head(10))
print('\nMissing data in test set: \n'  ,missing_data_test.head(10))

def find_uni(df):
    col_list = df.columns
    redundant_col =[]
    for col in col_list:
        if df[col].nunique() == 1:
            redundant_col.append(col)
    return redundant_col


redundant_col_train  = find_uni(train_df)
redundant_col_test = find_uni(test_df)

print ('Number of redundant features in train data :',len(redundant_col_train))
print ('Redundant Feature :', redundant_col_train)

print ('\n Number of redundant features in test data :',len(redundant_col_test))
print ('Redundant Feature :', redundant_col_test)
intersection = list(set(redundant_col_train) & set(redundant_col_test))

train_df.drop(intersection, axis =1, inplace = True)
test_df.drop(intersection, axis =1, inplace = True)
train_df.shape,test_df.shape
train_df.head()
import datetime

train_df['date'] = train_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
train_df['date'] = pd.to_datetime(train_df['date'])
print ('train_data:', train_df['date'].describe())

test_df['date'] = test_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
test_df['date'] = pd.to_datetime(test_df['date'])
print ('\n test data:', test_df['date'].describe())
temp = train_df['channelGrouping'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='channelGrouping')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = train_df['device.browser'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='device.browser')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = train_df['device.deviceCategory'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='device.deviceCategory')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
for col in ['visitNumber', 'totals.hits', 'totals.pageviews', 'totals.transactionRevenue']:
    train_df[col] = train_df[col].astype(float)
plt.hist(np.log(train_df.loc[train_df['totals.transactionRevenue'].isna() == False, 'totals.transactionRevenue']));
plt.title('Distribution of revenue');
