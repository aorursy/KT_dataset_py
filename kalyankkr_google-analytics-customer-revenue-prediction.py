# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as datetime
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
%matplotlib inline

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
from sklearn import metrics
import gc
gc.enable()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
json_data=["device","geoNetwork","totals","trafficSource"]

gc.enable()
features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',\
       'visitNumber', 'visitStartTime', 'device_browser',\
       'device_deviceCategory', 'device_isMobile', 'device_operatingSystem',\
       'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',\
       'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',\
       'geoNetwork_subContinent', 'totals_bounces', 'totals_hits',\
       'totals_newVisits', 'totals_pageviews', 'totals_transactionRevenue',\
       'trafficSource_adContent', 'trafficSource_campaign',\
       'trafficSource_isTrueDirect', 'trafficSource_keyword',\
       'trafficSource_medium', 'trafficSource_referralPath',\
       'trafficSource_source']
def load_df(csv_path='../input/train_v2.csv'):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    dfs = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                    chunksize = 100000)
    for df in dfs:
        df.reset_index(drop = True,inplace = True)
        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

        print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        use_df = df[features]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis = 0).reset_index(drop = True)
        print(ans.shape)
    return ans

train = load_df()
train.shape
train.head()
test=load_df(csv_path='../input/test_v2.csv')
train["totals_transactionRevenue"] = train["totals_transactionRevenue"].astype('float')
total_revenue=train.groupby("fullVisitorId")["totals_transactionRevenue"].sum().reset_index()
plt.figure(figsize=(8,6))
plt.scatter(range(total_revenue.shape[0]), np.sort(np.log1p(total_revenue["totals_transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()
def chats(data):
    trace = go.Bar(y=data.index[::-1],
                   x=data.values[::-1],
                   showlegend=False,
                   orientation = 'h',
    )
    return trace
data=train.groupby("device_browser")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace1=chats(data["count"].head(10))
trace2=chats(data["count of non-zero revenue"].head(10))
trace3=chats(data["mean"].head(10))


data=train.groupby("device_deviceCategory")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace4=chats(data["count"].head(10))
trace5=chats(data["count of non-zero revenue"].head(10))
trace6=chats(data["mean"].head(10))


data=train.groupby("device_operatingSystem")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace7=chats(data["count"].head(10))
trace8=chats(data["count of non-zero revenue"].head(10))
trace9=chats(data["mean"].head(10))


# Creating two subplots
fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 
                          subplot_titles=["Device Browser - Count", "Device Browser - Non-zero Revenue Count", "Device Browser - Mean Revenue",
                                          "Device Category - Count",  "Device Category - Non-zero Revenue Count", "Device Category - Mean Revenue", 
                                          "Device OS - Count", "Device OS - Non-zero Revenue Count", "Device OS - Mean Revenue"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots')
train.head(10)
#country
network_Country=train.groupby("geoNetwork_country")["totals_transactionRevenue"].agg(["size","count","mean"])
network_Country.columns=["Country_count", "Country_count of non-zero revenue", "mean"]
network_Country=network_Country.sort_values(by="Country_count",ascending=False)
trace1=chats(network_Country["Country_count"].head(10))
trace2=chats(network_Country["Country_count of non-zero revenue"].head(10))
trace3=chats(network_Country["mean"].head(10))



#continent
network_continent=train.groupby("geoNetwork_continent")["totals_transactionRevenue"].agg(["size","count","mean"])
network_continent.columns=["Continent_count", "Continent_count of non-zero revenue", "mean"]
network_continent=network_continent.sort_values(by="Continent_count",ascending=False)
trace4=chats(network_continent["Continent_count"].head(10))
trace5=chats(network_continent["Continent_count of non-zero revenue"].head(10))
trace6=chats(network_continent["mean"].head(10))


network_continent=train.groupby("geoNetwork_networkDomain")["totals_transactionRevenue"].agg(["size","count","mean"])
network_continent.columns=["networkDomain_count", "networkDomain_count of non-zero revenue", "mean"]
network_continent=network_continent.sort_values(by="networkDomain_count",ascending=False)
trace7=chats(network_continent["networkDomain_count"].head(10))
trace8=chats(network_continent["networkDomain_count of non-zero revenue"].head(10))
trace9=chats(network_continent["mean"].head(10))

fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 
                          subplot_titles=["Traffic Source - Count", "Traffic Source - Non-zero Revenue Count", "Traffic Source - Mean Revenue",
                                          "Traffic Source Medium - Count",  "Traffic Source Medium - Non-zero Revenue Count", "Traffic Source Medium - Mean Revenue"
                                          ])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig['layout'].update(height=1000, width=1200, paper_bgcolor='rgb(233,233,233)', title="Traffic Source Plots")
py.iplot(fig, filename='geoNetwork-plots')
#traffic_source
traffic_source=train.groupby("trafficSource_source")["totals_transactionRevenue"].agg(["size","count","mean"])
traffic_source.columns=["traffic_source_count", "traffic_source_count of non-zero revenue", "mean"]
traffic_source=traffic_source.sort_values(by="traffic_source_count",ascending=False)
trace1=chats(traffic_source["traffic_source_count"].head(10))
trace2=chats(traffic_source["traffic_source_count of non-zero revenue"].head(10))
trace3=chats(traffic_source["mean"].head(10))



#medium
traffic_medium=train.groupby("trafficSource_medium")["totals_transactionRevenue"].agg(["size","count","mean"])
traffic_medium.columns=["traffic_medium_count", "traffic_medium_count of non-zero revenue", "mean"]
traffic_medium=traffic_medium.sort_values(by="traffic_medium_count",ascending=False)
trace4=chats(traffic_medium["traffic_medium_count"].head(10))
trace5=chats(traffic_medium["traffic_medium_count of non-zero revenue"].head(10))
trace6=chats(traffic_medium["mean"].head(10))




fig = tools.make_subplots(rows=2, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 
                          subplot_titles=["Traffic Source - Count", "Traffic Source - Non-zero Revenue Count", "Traffic Source - Mean Revenue",
                                          "Traffic Source Medium - Count",  "Traffic Source Medium - Non-zero Revenue Count", "Traffic Source Medium - Mean Revenue"
                                          ])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)


fig['layout'].update(height=1000, width=1200, paper_bgcolor='rgb(233,233,233)', title="Traffic Source Plots")
py.iplot(fig, filename='geoNetwork-plots')
import datetime
def scatter_plot(data):
    trace = go.Scatter(
        x=data.index[::-1],
        y=data.values[::-1],
        showlegend=False,
        #mode = 'lines+markers',
    )
    return trace
train['date'] = train['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
date_trans=train.groupby("date")["totals_transactionRevenue"].agg(["size","count"])
date_trans.columns = ["count", "count of non-zero revenue"]
date_trans = date_trans.sort_index()

trace1=scatter_plot(date_trans["count"])
trace2=scatter_plot(date_trans["count of non-zero revenue"])
fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Date - Count", "Date - Non-zero Revenue count"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="date-plots for count & Revenue")
py.iplot(fig, filename='date-plots')
print("Variables in train and not in test` : ", set(train.columns).difference(set(test.columns)))
train.shape,test.shape
train["totals_transactionRevenue"].fillna(0, inplace=True)
train_y = train["totals_transactionRevenue"].values
num_cols = ["totals_hits", "totals_pageviews", "visitNumber", "visitStartTime", 'totals_bounces',  'totals_newVisits']    
for col in num_cols:
    train[col] = train[col].astype(float)
    test[col] = test[col].astype(float)
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime', 'nb_sessions', 'max_visits','visit_time','totals_visits'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]

categorical_features
train_val=np.log1p(train_y)
for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])
train.head()
train["totals_transactionRevenue"].fillna(0, inplace=True)
train_y = train["totals_transactionRevenue"]
train_val=np.log1p(train_y)
train_x,test_x,val_x,val_y=train_test_split(train,train_val,test_size=0.2, random_state=42)

trn_x=train_x[categorical_features+num_cols]
tst_x=test_x[categorical_features+num_cols]

test_X=test[categorical_features+num_cols]
test_X.head()
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(trn_x, val_x,tst_x, val_y, test_X)
from sklearn import metrics
pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":test_x["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = test_x["totals_transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
submission = pd.DataFrame({"fullVisitorId":test["fullVisitorId"].values})
pred_test[pred_test<0] = 0
submission["PredictedLogRevenue"] = np.expm1(pred_test)
submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
submission.columns = ["fullVisitorId", "PredictedLogRevenue"]
submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission.to_csv("predicted_values.csv", index=False)
submission.head(10)
fig, ax = plt.subplots(figsize=(18,20))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("Feature Importance", fontsize=15)
plt.show()
