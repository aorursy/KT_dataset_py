import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
# Main files
train = pd.read_csv("../input/train.csv",
                    dtype={'fullVisitorId': 'str'} # Important!!
                   )
test = pd.read_csv("../input/test.csv",
                  dtype={'fullVisitorId': 'str'}, # Important!!
                  )
print('Size of train data', train.shape)
print('Size of test data', test.shape)
train.shape
test.shape
train.head()
def string_to_dict(dict_string):
    # Convert to proper json format, from a string. json.load is for dictionary
    return json.loads(dict_string)

json_cols = ["device", "geoNetwork","totals","trafficSource"]


def create_new_dataset(df):
    new = df.copy()
    for col in json_cols:
        new[col] = new[col].apply(string_to_dict)
        #the following line will convert the JSON values into series and concatenat to the train dataframe
        new = pd.concat([new,(pd.io.json.json_normalize(new[col]))], axis=1)

    new = new.drop(["device", "geoNetwork","totals","trafficSource"], axis = 1)
    
    return new
new_train = create_new_dataset(train)
new_test = create_new_dataset(test)
new_train.info()
pd.value_counts(new_train.dtypes).plot(kind="bar")
plt.title("type of train data")
new_train.describe()
# Check for duplicates
idsUnique = len(set(new_train.sessionId ))
idsTotal = new_train.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate sessionId for " + str(idsTotal) + " total entries")
#Check Unique values
constant_column = [col for col in new_train.columns if len(new_train[col].unique()) == 1]
print(list(constant_column))

new_train.drop(columns=constant_column,inplace=True)
new_test.drop(columns=constant_column,inplace=True)

print(f'Dropped {len(constant_column)} columns.')
def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['visitStartTime_'] = pd.to_datetime(df['visitStartTime'],unit="s")
    df['visitStartTime_year'] = df['visitStartTime_'].apply(lambda x: x.year)
    df['visitStartTime_month'] = df['visitStartTime_'].apply(lambda x: x.month)
    df['visitStartTime_day'] = df['visitStartTime_'].apply(lambda x: x.day)
    df['visitStartTime_weekday'] = df['visitStartTime_'].apply(lambda x: x.weekday())
    return df
date_features = [#"year","month","day","weekday",'visitStartTime_year',
    "visitStartTime_month","visitStartTime_day","visitStartTime_weekday"]
add_time_features(new_train)
add_time_features(new_test)
#Target data is non numerical and has NaN. Remember our target variable has to be the log
new_train.transactionRevenue = new_train.transactionRevenue.astype(float)
target= np.log(new_train.transactionRevenue.dropna())

#Histogram

plt.figure(figsize=(12,7))
sns.distplot(target);
plt.xlabel("Revenue")
plt.ylabel("Number of Buyers")
plt.title("Target Distribution")

print ("Skew is:", target.skew())
print("Kurtosis: %f" % target.kurt())
grouped = new_train.groupby('fullVisitorId')["transactionRevenue"].sum().reset_index()
target_group = np.log(grouped.loc[grouped['transactionRevenue'] > 0, 'transactionRevenue'])

plt.figure(figsize=(12,7))
sns.distplot(target_group);
plt.xlabel("Revenue")
plt.ylabel("Number of Buyers")
plt.title("Target Distribution")

print ("Skew is:", target_group.skew())
print("Kurtosis: %f" % target_group.kurt())
#https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue

nzi = pd.notnull(new_train["transactionRevenue"]).sum()
nzr = (grouped["transactionRevenue"]>0).sum()
print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / new_train.shape[0])
print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / grouped.shape[0])
numeric_features = new_train.select_dtypes(include=[np.number])
numeric_features.dtypes
#Original code from : https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue

from plotly import tools

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

cnt_srs = new_train.groupby('date')['transactionRevenue'].agg(['size', 'count'])
cnt_srs.columns = ["count", "count of non-zero revenue"]
cnt_srs = cnt_srs.sort_index()
#cnt_srs.index = cnt_srs.index.astype('str')
trace1 = scatter_plot(cnt_srs["count"], 'red')
trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')

fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Date - Count", "Date - Non-zero Revenue count"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')

new_train.visitId.value_counts()[:10]
new_train.visitNumber.value_counts()[:10]
new_train.visitStartTime.value_counts()[:10]
categorical_features = new_train.select_dtypes(exclude=[np.number])
categorical_features.dtypes
sns.countplot(new_train.channelGrouping,order=new_train.channelGrouping.value_counts().iloc[0:10].index)
plt.xticks(rotation=90);
new_train.fullVisitorId.value_counts()[:10]
new_train.sessionId.value_counts()[:10]
plt.figure(figsize=(12,7))
sns.countplot(new_train.browser,order=new_train.browser.value_counts().iloc[0:10].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.deviceCategory)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.isMobile)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.operatingSystem,order=new_train.operatingSystem.value_counts().iloc[0:10].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.city,order=new_train.city.value_counts().iloc[:50].index)
plt.xticks(rotation=90);

plt.figure(figsize=(18,7))
sns.countplot(new_train.city,order=new_train.city.value_counts().iloc[1:100].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.continent,order=new_train.continent.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.country,order=new_train.country.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(18,7));
sns.countplot(new_train.metro,order=new_train.metro.value_counts().iloc[:50].index);
plt.xticks(rotation=90);
plt.figure(figsize=(18,7));
sns.countplot(new_train.metro,order=new_train.metro.value_counts().iloc[2:50].index);
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.networkDomain,order=new_train.networkDomain.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.region,order=new_train.region.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.subContinent,order=new_train.subContinent.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
sns.countplot(new_train.bounces)
plt.xticks(rotation=90);
new_train.bounces.value_counts()
new_train.bounces.isnull().sum()
plt.figure(figsize=(12,7))
sns.countplot(new_train.hits,order=new_train.hits.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.newVisits,order=new_train.newVisits.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
new_train.newVisits.value_counts()
new_train.newVisits.isnull().sum()
plt.figure(figsize=(12,7))
sns.countplot(new_train.pageviews,order=new_train.pageviews.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.adContent,order=new_train.adContent.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train["adwordsClickInfo.adNetworkType"])
plt.xticks(rotation=90);
new_train["adwordsClickInfo.adNetworkType"].value_counts()
new_train["adwordsClickInfo.adNetworkType"].isnull().sum()
plt.figure(figsize=(12,7))
sns.countplot(new_train["adwordsClickInfo.gclId"],order=new_train["adwordsClickInfo.gclId"].value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train["adwordsClickInfo.isVideoAd"],order=new_train["adwordsClickInfo.isVideoAd"].value_counts().iloc[:50].index)
plt.xticks(rotation=90);
new_train["adwordsClickInfo.isVideoAd"].value_counts()
new_train["adwordsClickInfo.isVideoAd"].isnull().sum()
plt.figure(figsize=(12,7))
sns.countplot(new_train["adwordsClickInfo.page"],order=new_train["adwordsClickInfo.page"].value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train["adwordsClickInfo.slot"],order=new_train["adwordsClickInfo.slot"].value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.campaign,order=new_train.campaign.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.campaignCode,order=new_train.campaignCode.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
new_train.campaignCode.value_counts()
new_train.campaignCode.isnull().sum()
plt.figure(figsize=(12,7))
sns.countplot(new_train.isTrueDirect,order=new_train.isTrueDirect.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
new_train.isTrueDirect.value_counts()
new_train.isTrueDirect.isnull().sum()
plt.figure(figsize=(12,7))
sns.countplot(new_train.keyword,order=new_train.keyword.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.medium,order=new_train.medium.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.referralPath,order=new_train.referralPath.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.source,order=new_train.source.value_counts().iloc[:50].index)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.year)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.month)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.day)
plt.xticks(rotation=90);
plt.figure(figsize=(12,7))
sns.countplot(new_train.weekday)
plt.xticks(rotation=90);
#{0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
def horizontal_bar_chart(cnt_srs, color):
    trace = go.Bar(
        y=cnt_srs.index[::-1],
        x=cnt_srs.values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace
cnt_srs = new_train.groupby('visitNumber')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace5 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["VisitNumber - Count",
                                          "VisitNumber - Non-zero Revenue Count", 
                                          "VisitNumber - Mean Revenue"]);

fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');
cnt_srs = new_train.groupby('channelGrouping')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace5 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["Channel Grouping - Count",
                                          "Channel Grouping - Non-zero Revenue Count", 
                                          "Channel Grouping - Mean Revenue"]);

fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');
# Device Browser

#size includes NaN values, count does not:
cnt_srs = new_train.groupby('browser')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["Device Browser - Count",
                                          "Device Browser - Non-zero Revenue Count", 
                                          "Device Browser - Mean Revenue"]);

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');
cnt_srs = new_train.groupby('deviceCategory')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace5 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["Device Category - Count",
                                          "Device Category - Non-zero Revenue Count", 
                                          "Device Category - Mean Revenue"]);

fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');

cnt_srs = new_train.groupby('operatingSystem')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace5 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["Device OS - Count",
                                          "Device OS - Non-zero Revenue Count", 
                                          "Device OS - Mean Revenue"]);

fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');
cnt_srs = new_train.groupby('pageviews')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace5 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["Page Views - Count",
                                          "Page Views - Non-zero Revenue Count", 
                                          "Page Views - Mean Revenue"]);

fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');
cnt_srs = new_train.groupby('continent')['transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace5 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

fig = tools.make_subplots(rows=1, cols=3, 
                          vertical_spacing=0.04, 
                          subplot_titles=["Continent - Count",
                                          "Continent - Non-zero Revenue Count", 
                                          "Continent - Mean Revenue"]);

fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 1, 3)
fig['layout'].update(height=600, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
py.iplot(fig, filename='device-plots');