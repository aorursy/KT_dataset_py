
import os
print(os.listdir("../input"))
import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
import feather as fe
warnings.filterwarnings('ignore')

from pandas.io.json import json_normalize

import json

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import Imputer

from sklearn import preprocessing

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv("../input/ga-customer-revenue-prediction/train.csv",sep=',')

test_df = pd.read_csv("../input/ga-customer-revenue-prediction/test.csv",sep=',')
## This method of flattening the JSON columns is a very popular approach obtained from the Kaggle discussion forums
json_columns = ['device', 'geoNetwork','totals', 'trafficSource']
def load_df(filename):
    path = "../input/ga-customer-revenue-prediction/" + filename
    df = pd.read_csv(path, converters={column: json.loads for column in json_columns}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in json_columns:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
train_df = load_df("train.csv")
test_df = load_df("test.csv")

train_df.to_feather('train.feather')
test_df.to_feather('test.feather')
train_df = pd.read_feather('train.feather')
test_df = pd.read_feather('test.feather')
count_row = train_df.shape[0]  
count_col = train_df.shape[1]
print("For Train : ")
print(count_row , count_col)

count_row = test_df.shape[0]  
count_col = test_df.shape[1]
print("For Test : ")
print(count_row , count_col)

print("*****************************")

# train_df=train_df.dropna()

# print("After removing NaN")
# count_row = train_df.shape[0]  
# count_col = train_df.shape[1]
# print(count_row , count_col)

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# colorscale=[[0.0, 'rgb(165,0,38)'], [0.005, 'rgb(215,48,39)'], [0.01, 'rgb(244,109,67)'], [0.02, 'rgb(253,174,97)'], [0.04, 'rgb(254,224,144)'], [0.05, 'rgb(224,243,248)'], [0.1, 'rgb(171,217,233)'], [0.25, 'rgb(116,173,209)'], [0.5, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
colorscale = [[0, 'rgb(102,194,165)'], [0.0007, 'rgb(102,194,165)'], 
              [0.004, 'rgb(132,200,165)'],[0.009, 'rgb(192,200,165)'],
              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 
              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 
              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = dict(type = 'choropleth', 
           locations = train_df["geoNetwork_country"].value_counts().index,
           locationmode = 'country names',
            colorscale = colorscale,
           z = train_df['totals_hits'].value_counts().values, 
           text = train_df["geoNetwork_country"].value_counts().index,
           colorbar = {'title':'Total Hits '})
layout = dict(title = 'Hits', 
              height = 1000,
              geo = dict(showframe = False, 
                       projection = {'type': 'mercator'})
        )
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)

train_corr = train_df.copy()
plt.figure(figsize=(15,15))
sns.heatmap(train_corr.corr(method="pearson"), annot=True, cmap="YlGnBu")
train_corr.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('Accent'), axis=1)
import datetime

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

def add_date_features(df):
    df['date'] = df['date'].astype(str)
    df["date"] = df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    df["date"] = pd.to_datetime(df["date"])
    
    df["month"]   = df['date'].dt.month
    df["day"]     = df['date'].dt.day
    df["weekday"] = df['date'].dt.weekday
    return df 

# train_df['date'] = train_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
train_df = add_date_features(train_df)
cnt_srs = train_df.groupby('date')['totals_transactionRevenue'].agg(['size', 'count'])
cnt_srs.columns = ["count", "count of non-zero revenue"]
cnt_srs = cnt_srs.sort_index()
#cnt_srs.index = cnt_srs.index.astype('str')


import plotly.offline as py
from plotly import tools
trace = scatter_plot(cnt_srs["count"], 'black')
py.init_notebook_mode(connected=True)
fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.08,subplot_titles=["Date - Count"])
fig.append_trace(trace, 1, 1)
py.iplot(fig, filename='date-plots')
t = train_df['channelGrouping'].value_counts()
values1 = t.values 
index1 = t.index
domain1 = {'x': [0.2, 0.50], 'y': [0.0, 0.33]}
fig = {
  "data": [
    {
      "values": values1,
      "labels": index1,
      "domain": {"x": [0, .48]},
    "marker" : dict(colors=["#y88b3c" ,'#cb27fb',  '#b1b1b2']),
      "name": "Channel Grouping",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }
   ],
  "layout": {"title":"Channel Grouping",
      "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": " ",
                "x": 0.11,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)
train_df["totals_transactionRevenue"] = train_df["totals_transactionRevenue"].astype('float')
continent = train_df.groupby('geoNetwork_continent')['totals_transactionRevenue'].agg(['size', 'count', 'sum'])
continent.columns = ["total count", "count of non-zero revenue", "sum"]
continent.head()
print(continent['total count'])
continent['total count'] = np.log(continent['total count'])
continent.groupby('geoNetwork_continent')['total count'].mean().sort_index().plot.bar(color = 'b');

plt.title('Log(Total Count) vs Continent');
plt.ylabel('Log(Total Count');
plt.xlabel('Continent')

print (continent['count of non-zero revenue'])
# continent['count of non-zero revenue'] = np.log(continent['count of non-zero revenue'])
continent['count of non-zero revenue'] = np.log(continent['count of non-zero revenue'])
continent.groupby('geoNetwork_continent')['count of non-zero revenue'].mean().plot.bar(color = 'b');
plt.title('Log(Total Non-Zero Revenue Count) vs Continent');
plt.ylabel('Log(Non Zero Revenue Count)');
plt.xlabel('Continent')
print(continent['sum'])
continent['sum'] = np.log(continent['sum'])
continent.groupby('geoNetwork_continent')['sum'].mean().sort_index().plot.bar(color = 'b')
plt.title('Log(Total  Revenue) vs Continent')
plt.ylabel('Revenue')
plt.xlabel('Continent')
train_df["totals_transactionRevenue"] = train_df["totals_transactionRevenue"].astype('float')
subCont = train_df.groupby('geoNetwork_subContinent')['totals_transactionRevenue'].agg(['size', 'count', 'sum'])
subCont.columns = ["total count", "count of non-zero revenue", "sum"]
print (subCont)
print(subCont['total count'])
subCont['total count'] = np.log(subCont['total count'])
subCont.groupby('geoNetwork_subContinent')['total count'].mean().sort_index().plot.bar(color = 'b')
plt.title('Log( Total Count ) vs Sub-Continent')
plt.ylabel('Total Count')
plt.xlabel('Sub-Continent')
subCont['count of non-zero revenue'] = np.log(subCont['count of non-zero revenue'])
print (subCont['count of non-zero revenue'])
subCont.groupby('geoNetwork_subContinent')['count of non-zero revenue'].mean().plot.bar(color = 'b')
plt.title('Log(Total Non-Zero Revenue Count) vs Sub-Continent')
plt.ylabel('Log(Total Non-Zero Revenue Count)')
plt.xlabel('Sub-Continent')
print(subCont['sum'])
subCont['sum'] = np.log(subCont['sum'])
subCont.groupby('geoNetwork_subContinent')['sum'].mean().sort_index().plot.bar(color = 'b')
plt.title('Log(Total  Revenue) vs Sub-Continent')
plt.ylabel('Revenue')
plt.xlabel('Sub-Continent')
geo_cols = ["geoNetwork_city", "geoNetwork_country", "geoNetwork_subContinent", "geoNetwork_continent"]
colors = ["#008080"]
traces = []
for i, col in enumerate(geo_cols):
    t = train_df[col].value_counts()
    traces.append(go.Bar(marker=dict(color=colors[0]),orientation="h", y = t.index[:15], x = t.values[:15]))

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=["Visits: City", "Visits: Country","Visits: Sub Continent","Visits: Continent"], print_grid=False)
fig.append_trace(traces[0], 1, 1)
fig.append_trace(traces[1], 1, 2)
fig.append_trace(traces[2], 2, 1)
fig.append_trace(traces[3], 2, 2)
fig['layout'].update(height=600,width=1000, showlegend=False)
iplot(fig)


train_df["totals_transactionRevenue"] = train_df["totals_transactionRevenue"].astype('float')
userCount_df = train_df.groupby('fullVisitorId')['fullVisitorId'].agg(['size'])
userCount_df=userCount_df.sort_values(by=['size'],ascending=False)
print(userCount_df.head(10))

userAmount_df = train_df.groupby('fullVisitorId')['totals_transactionRevenue'].agg(['count','mean'])
userAmount_df=userAmount_df.sort_values(by=['count'],ascending=False)
print(userAmount_df.head(10))

totalTransaction = userCount_df['size'].sum()
print(totalTransaction)

userCount_df['buying probability'] = userCount_df['size']/totalTransaction
userCount_df=userCount_df.sort_values(by=['buying probability'],ascending=False)
print(userCount_df.head(10))

probSum = userCount_df['buying probability'].sum()
print(probSum)
train_df = train_df.drop('trafficSource_campaignCode',1)
const_cols = []
for col in train_df.columns:
    if len(train_df[col].value_counts()) == 1:
        const_cols.append(col)

## non relevant columns
non_relevant = ["visitNumber", "date", "fullVisitorId", "sessionId", "visitId", "visitStartTime"]
test_df = add_date_features(test_df)
### The label encoding part is inspired from another Kernel :

from sklearn.preprocessing import LabelEncoder

categorical_columns = [c for c in train_df.columns if not c.startswith("total")]
categorical_columns = [c for c in categorical_columns if c not in const_cols + non_relevant]
for c in categorical_columns:

    le = LabelEncoder()
    train_vals = list(train_df[c].values.astype(str))
    test_vals = list(test_df[c].values.astype(str))
    
    le.fit(train_vals + test_vals)
    
    train_df[c] = le.transform(train_vals)
    test_df[c] = le.transform(test_vals)
def normalize_numerical_columns(df, isTrain = True):
    df["totals_hits"] = df["totals_hits"].astype(float)
    df["totals_hits"] = (df["totals_hits"] - min(df["totals_hits"])) / (max(df["totals_hits"]) - min(df["totals_hits"]))

    df["totals_pageviews"] = df["totals_pageviews"].astype(float)
    df["totals_pageviews"] = (df["totals_pageviews"] - min(df["totals_pageviews"])) / (max(df["totals_pageviews"]) - min(df["totals_pageviews"]))
    
    if isTrain:
        df["totals_transactionRevenue"] = df["totals_transactionRevenue"].fillna(0.0)
    return df 
train_df = normalize_numerical_columns(train_df)
test_df  = normalize_numerical_columns(test_df, isTrain = False)
from sklearn.model_selection import train_test_split
import lightgbm as lgb 
features = [c for c in train_df.columns if c not in const_cols + non_relevant]
features.remove("totals_transactionRevenue")

train_df_new["totals_transactionRevenue"] = np.log1p(train_df["totals_transactionRevenue"].astype(float))
train_x, valid_x, train_y, valid_y = train_test_split(train_df[features], train_df_new["totals_transactionRevenue"], test_size=0.2, random_state=20)


lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}
    
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_val = lgb.Dataset(valid_x, label=valid_y)
model = lgb.train(lgb_params, lgb_train, 200, valid_sets=[lgb_val], early_stopping_rounds=150, verbose_eval=20)



preds = model.predict(test_df[features], num_iteration=model.best_iteration)
test_df["PredictedLogRevenue"] = np.expm1(preds)
submission = test_df.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0)
submission.to_csv("Output_7.csv", index=False)
submission.head()
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
train_df_copy = train_df.copy()
from sklearn import metrics
count = 0
bestValue = 1.65553
def ptest(train_df_copy):
    
    train_df_new = pd.DataFrame()
    train_df_new["totals_transactionRevenue"] = np.log1p(train_df_copy["totals_transactionRevenue"].astype(float))
    train_x, valid_x, train_y, valid_y = train_test_split(train_df_copy[features], train_df_new["totals_transactionRevenue"], test_size=0.2, random_state=20)


    lgb_params = {"objective" : "regression", "metric" : "rmse",
                  "num_leaves" : 50, "learning_rate" : 0.02, 
                  "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}


    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_val = lgb.Dataset(valid_x, label=valid_y)
    model = lgb.train(lgb_params, lgb_train, 200, valid_sets=[lgb_val], early_stopping_rounds=100, verbose_eval=200)
    
    pred1 = model.predict(valid_x, num_iteration=model.best_iteration)
    rmse = np.sqrt(metrics.mean_squared_error(pred1, valid_y))    
    return rmse

testing_cols = ['trafficSource_campaign' , 'totals_hits' , 'totals_pageviews']
for cols in testing_cols:
    
    count = 0
    avg = 0 
    for i in range (1,100):
        print(i)
        train_df_copy[cols] = np.random.permutation(train_df_copy[cols])
        rmse =  ptest(train_df_copy)   
        if rmse<bestValue:
            count+=1
    print(count)
    print("For ",cols," number of times RMSE is lesser than base is : ",count)
 