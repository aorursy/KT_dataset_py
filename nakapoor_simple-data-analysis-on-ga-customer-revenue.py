import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()

%matplotlib inline
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from IPython.display import HTML, display
import tabulate
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
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
%%time
train_df = load_df()
test_df = load_df("../input/test.csv")
print('size of training data : ', train_df.shape)
print('size of testing data  : ', test_df.shape)
train_df.head()
test_df.head()
total = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum() / train_df.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)


del train_df['trafficSource.campaignCode']
train_df['totals.transactionRevenue'].fillna(0, inplace=True)
train_df['trafficSource.adwordsClickInfo.page'].fillna(-99999, inplace=True)
test_df['trafficSource.adwordsClickInfo.page'].fillna(-99999, inplace=True)
print(train_df['trafficSource.adwordsClickInfo.page'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.page'].value_counts())
train_df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('Others', inplace=True)
test_df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('Others', inplace=True)
train_df['trafficSource.adwordsClickInfo.adNetworkType'] = np.where(train_df['trafficSource.adwordsClickInfo.adNetworkType'] != 'Google Search' , 'Others',train_df['trafficSource.adwordsClickInfo.adNetworkType'])
test_df['trafficSource.adwordsClickInfo.adNetworkType'] = np.where(test_df['trafficSource.adwordsClickInfo.adNetworkType'] != 'Google Search'  , 'Others',test_df['trafficSource.adwordsClickInfo.adNetworkType'])
print(train_df['trafficSource.adwordsClickInfo.adNetworkType'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.adNetworkType'].value_counts())
train_df['trafficSource.adwordsClickInfo.slot'].fillna('NA', inplace=True)
test_df['trafficSource.adwordsClickInfo.slot'].fillna('NA', inplace=True)
#train_df['trafficSource.adwordsClickInfo.slot'] = np.where(train_df['trafficSource.adwordsClickInfo.slot'] != ["RHS", "Top"] , 'NA',train_df['trafficSource.adwordsClickInfo.slot'])
test_df['trafficSource.adwordsClickInfo.slot'] = np.where(test_df['trafficSource.adwordsClickInfo.slot'] ==  "Google Display Network" , 'NA',test_df['trafficSource.adwordsClickInfo.slot'])
print(train_df['trafficSource.adwordsClickInfo.slot'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.slot'].value_counts())
train_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
test_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
print(train_df['trafficSource.adwordsClickInfo.isVideoAd'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.isVideoAd'].value_counts())
train_df['trafficSource.isTrueDirect'].fillna(False, inplace=True)
test_df['trafficSource.isTrueDirect'].fillna(False, inplace=True)
print(train_df['trafficSource.isTrueDirect'].value_counts())
print(test_df['trafficSource.isTrueDirect'].value_counts())

train_df['totals.bounces'].fillna(0, inplace=True)
test_df['totals.bounces'].fillna(0, inplace=True)
print(train_df['totals.bounces'].value_counts())
print(test_df['totals.bounces'].value_counts())
train_df['totals.newVisits'].fillna(0, inplace=True)
test_df['totals.newVisits'].fillna(0, inplace=True)
print(train_df['totals.newVisits'].value_counts())
print(test_df['totals.newVisits'].value_counts())
total = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum() / train_df.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.loc[missing_application_train_data['Percent'] > 0]

del missing_application_train_data
feats_counts = train_df.nunique(dropna = False).sort_values(ascending = False)
values = feats_counts.values
trace1 = go.Bar(
    x = feats_counts.index,
    y = values ,
)

data = [trace1]

layout = go.Layout(
    title = "# of unique values in each column in dataframe",
    xaxis=dict(
        title='Features Names',
        domain=[0, 0.5]
    ),
    
    yaxis=dict(
        title='# of unique constant values'
        
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Campaign code')

constant_features = feats_counts.loc[feats_counts==1].index.tolist()
print (constant_features)
train_df.drop(constant_features,axis = 1,inplace=True)
test_df.drop(constant_features,axis = 1,inplace=True)
del constant_features
feats_counts = train_df.nunique(dropna = False).sort_values(ascending = False)
print(feats_counts)
del feats_counts
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
revnSum = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().sort_values(ascending = True).reset_index()
#revnSum = np.log1p(revnSum['totals.transactionRevenue']/1000000)
revnSum = np.log1p(revnSum['totals.transactionRevenue'])
plt.figure(figsize=(8,6))
plt.scatter(revnSum.index, revnSum.values)
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue for visitors', fontsize=12)
plt.show()
train_df['if_TransRev'] = np.where(train_df['totals.transactionRevenue'] > 0.0 , 1,0)
feats_counts = train_df['if_TransRev'].value_counts()
values = (feats_counts/feats_counts.sum())*100
trace1 = go.Bar(
    x = feats_counts.index,
    y = values ,
)

data = [trace1]

layout = go.Layout(
    title = "% of visitors with transaction vs Non transaction",
    xaxis=dict(
        title='TransactionRevenue',
        domain=[0, 0.5]
    ),
    
    yaxis=dict(
        title='% of visitors'
        
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Campaign code')

#train_df.to_csv("filterTrain_df.csv", index=False)
tempTrain_df = train_df[train_df['totals.transactionRevenue'] > 0.0]
tempTrain_df['device.isMobile'] = np.where(tempTrain_df['device.isMobile'] == True , 1,0)
tempTrain_df['totals.newVisits'] = np.where(tempTrain_df['totals.newVisits'] == True , 1,0)
tempTrain_df['trafficSource.adwordsClickInfo.isVideoAd'] = np.where(tempTrain_df['trafficSource.adwordsClickInfo.isVideoAd'] == True , 1,0)
tempTrain_df['trafficSource.isTrueDirect'] = np.where(tempTrain_df['trafficSource.isTrueDirect'] == True , 1,0)
tempTrain_df['totals.bounces'] = np.where(tempTrain_df['totals.bounces'] == True , 1,0)
tempTrain_df.shape
def bar_chart(lables, values):
    trace = go.Bar(
        x=lables,
        y=values,
        showlegend=False,
        marker=dict(
            color='rgba(28,32,56,0.84)',
        )
    )
    return trace

feats_counts = tempTrain_df['device.isMobile'].value_counts()
trace1 = bar_chart(lables = feats_counts.index, values = (feats_counts/feats_counts.sum())*100)

feats_counts = tempTrain_df['trafficSource.adwordsClickInfo.isVideoAd'].value_counts()
trace2 = bar_chart(lables = feats_counts.index, values = (feats_counts/feats_counts.sum())*100)

feats_counts = tempTrain_df['totals.newVisits'].value_counts()
trace3 = bar_chart(lables = feats_counts.index, values = (feats_counts/feats_counts.sum())*100)

feats_counts = tempTrain_df['trafficSource.isTrueDirect'].value_counts()
trace4 = bar_chart(lables = feats_counts.index, values = (feats_counts/feats_counts.sum())*100)

feats_counts = tempTrain_df['totals.bounces'].value_counts()
trace5 = bar_chart(lables = feats_counts.index, values = (feats_counts/feats_counts.sum())*100)

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.06, 
                          subplot_titles=["Mobile Ads", "Video Ads","New Visit", "Direct Visit","if user bounced"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)


fig['layout'].update(height=1200, width=800, paper_bgcolor='rgb(233,233,233)', title="Impact of 5 binary features on transactions with revenue")

py.iplot(fig, filename='plots_2')

