# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from datetime import datetime
# Plotly
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
print(plotly.__version__)
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
#plot_bgcolor="#f5efe3"
#plot_bgcolor="#faf7e6"
#plot_bgcolor="#faf6e9"
#plot_bgcolor="#fffdf6"
plot_bgcolor = "#fcfaf1"
paper_bgcolor = plot_bgcolor
%%time
train_df = pd.read_feather("../input/data-preprocessing/train_preprocessed.feather")
%%time
test_df = pd.read_feather("../input/data-preprocessing/test_preprocessed.feather")
train_df.columns
train_df.info()
train_df.shape
test_df.shape
train_df.head(10)
feats_to_examine = ["device.browser",
                "device.browserSize",
                "device.browserVersion",
                "device.deviceCategory",
                "device.flashVersion",
                "device.isMobile",
                "device.language",
                "device.mobileDeviceBranding",
                "device.mobileDeviceInfo",
                "device.mobileDeviceMarketingName",
                "device.mobileDeviceModel",
                "device.mobileInputSelector",
                "device.operatingSystem",
                "device.operatingSystemVersion",
                "device.screenColors",
                "device.screenResolution"]
unique_values_per_feature = []
num_unique_values = []
for fname in feats_to_examine: 
    unique_vals = train_df[fname].unique()
    unique_values_per_feature.append(':'.join([str(v) for v in unique_vals]))
    num_unique_values.append(len(unique_vals))
unique_feats_df = pd.DataFrame({'feature': feats_to_examine, 'unique_values': unique_values_per_feature, 'unique_values_count': num_unique_values})
unique_feats_df.sort_values(by=["unique_values_count"], ascending=False)
feats_to_examine = ["geoNetwork.city",
                "geoNetwork.cityId",
                "geoNetwork.continent",
                "geoNetwork.country",
                "geoNetwork.latitude",
                "geoNetwork.longitude",
                "geoNetwork.metro",
                "geoNetwork.networkDomain",
                "geoNetwork.networkLocation",
                "geoNetwork.region",
                "geoNetwork.subContinent"]
unique_values_per_feature = []
num_unique_values = []
for fname in feats_to_examine: 
    unique_vals = train_df[fname].unique()
    unique_values_per_feature.append(':'.join([str(v) for v in unique_vals]))
    num_unique_values.append(len(unique_vals))
unique_feats_df = pd.DataFrame({"feature": feats_to_examine, "unique_values": unique_values_per_feature, "unique_values_count": num_unique_values})
unique_feats_df.sort_values(by=["unique_values_count"], ascending=False)
feats_to_examine = train_df.columns
unique_values_per_feature = []
num_unique_values = []
for fname in feats_to_examine: 
    unique_vals = train_df[fname].unique()
    unique_values_per_feature.append(':'.join([str(v) for v in unique_vals]))
    num_unique_values.append(len(unique_vals))
unique_feats_df = pd.DataFrame({"feature": feats_to_examine, "unique_values": unique_values_per_feature, "unique_values_count": num_unique_values})
unique_feats_df = unique_feats_df.sort_values(by=["unique_values_count"], ascending=False)

# Plot
trace = go.Bar(y=unique_feats_df["unique_values_count"], 
               x=unique_feats_df["feature"], marker=dict(color="#fa360a"))
data = [trace]
layout = go.Layout(title="Count of unique values per feature",
                   xaxis=dict(title="Feature", tickfont=dict(size=8, color="grey")), 
                   yaxis=dict(type="log", title="# of unique values", tickfont=dict(size=8, color="grey"), showgrid=False), 
                   plot_bgcolor=plot_bgcolor, 
                   paper_bgcolor=paper_bgcolor, height=700)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
unique_feats_df
feats_to_examine = unique_feats_df[unique_feats_df["unique_values_count"] > 1].sort_index()["feature"].values
dataset_size = train_df.shape[0]
missing_percentage = [100.0 * train_df[fname].isna().sum() / dataset_size for fname in feats_to_examine]
missing_percentage_df = pd.DataFrame({"feature": feats_to_examine, "missing_percentage": missing_percentage})
missing_percentage_df = missing_percentage_df.sort_values(by=["missing_percentage"], ascending=False)
trace = go.Bar(y=missing_percentage_df["missing_percentage"], 
               x=missing_percentage_df["feature"], marker=dict(color="#fa360a"))
data = [trace]
layout = go.Layout(title="Missing data percentage of {} features".format(len(feats_to_examine)), 
                   xaxis=dict(title="Feature", tickfont=dict(size=8, color="grey")), 
                   yaxis=dict(title="Missing percentage (%)", tickfont=dict(size=8, color="grey"), showgrid=False), 
                   plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
useful_features = unique_feats_df[unique_feats_df["unique_values_count"] > 1].sort_index()["feature"].values.tolist()
useful_features
train_df[useful_features].info()
missing_percentage_df[missing_percentage_df["missing_percentage"] > 90]
useful_features.remove("trafficSource.campaignCode")
print("Number of useful features: {}".format(len(useful_features)))
# Helper functions
def replace_with_0(val):
    if not val:
        return 0
    return val

def to_yes_or_no(bool_val):
    if bool_val:
        return "yes"
    return "no"
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].fillna(0.0)
train_df[["totals.transactionRevenue"]].head()
train_df[["device.isMobile"]].groupby("device.isMobile")["device.isMobile"].count()
train_df["device.isMobile"] = train_df["device.isMobile"].apply(to_yes_or_no)
train_df[["device.isMobile"]].groupby("device.isMobile")["device.isMobile"].count()
train_df[["device.isMobile"]].head()
def convert_to_datetime(yyyymmdd):
    return datetime.strptime(yyyymmdd, "%Y%m%d")
train_df["date"] = train_df["date"].apply(str).apply(convert_to_datetime)
train_df["date"].head()
train_df["date.year"] = train_df["date"].apply(lambda x: x.year)
train_df["date.month"] = train_df["date"].apply(lambda x: x.month)
train_df["date.day"] = train_df["date"].apply(lambda x: x.day)
train_df["visitStartTime"] = train_df["visitStartTime"].apply(datetime.fromtimestamp)
train_df["visitStartTime.year"] = train_df["visitStartTime"].apply(lambda x: x.year)
train_df["visitStartTime.month"] = train_df["visitStartTime"].apply(lambda x: x.month)
train_df["visitStartTime.day"] = train_df["visitStartTime"].apply(lambda x: x.day)
train_df["visitStartTime.hour"] = train_df["visitStartTime"].apply(lambda x: x.hour)
train_df["visitStartTime.minute"] = train_df["visitStartTime"].apply(lambda x: x.minute)
train_df["totals.newVisits"] = train_df["totals.newVisits"].apply(replace_with_0)
train_df["totals.newVisits"].value_counts()
train_df["totals.bounces"] = train_df["totals.bounces"].apply(replace_with_0)
train_df["totals.bounces"].value_counts()
train_df["totals.transactionRevenue"].astype("float32").describe()
num_zero_revenues_rows = sum(train_df["totals.transactionRevenue"].astype("float32") == 0)
num_revenues_rows = sum(train_df["totals.transactionRevenue"].astype("float32") > 0)
trace = go.Bar(x=["Zero revenues", "Non-zero revenues"], 
               y=[num_zero_revenues_rows, num_revenues_rows], 
               marker=dict(color="#fa360a"))
data = [trace]
layout = go.Layout(title="Training data: count of zero vs non-zero revenues", 
                   xaxis=dict(title="totals.transactionRevenue", tickfont=dict(color="grey")), 
                   yaxis=dict(type="log", title="Number of data points", showgrid=False, tickfont=dict(color="grey")), 
                   plot_bgcolor=plot_bgcolor, 
                   paper_bgcolor=paper_bgcolor)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace = go.Histogram(x=train_df[train_df["totals.transactionRevenue"].astype("float32")>0]["totals.transactionRevenue"].astype("float32").apply(np.log), 
                     marker=dict(color="#fa360a"))
data = [trace]
layout = go.Layout(title="Distribution of transaction revenue", 
                   xaxis=dict(title="np.log(df['totals.transactionRevenue'])", tickfont=dict(color="grey")),
                   yaxis=dict(title="#count", showgrid=False, showline=False), 
                   plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
def generate_subplot_titles(features):
    subplot_titles = []
    for f in features:
        subplot_titles.append(f + " (all train)")
        subplot_titles.append(f + " (revenue > 0)")
    return subplot_titles
def plot_feature_counts(feats_to_plot, subplot_titles, log_yaxis=None):
    nrows = len(feats_to_plot)
    ncols = 2
    height = nrows * 300
    width = 900
    font_size = 8
    fig = tools.make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)
    for i, v in enumerate(feats_to_plot):
        count_feature_vals1 = train_df[v].value_counts()
        trace1 = go.Bar(x=count_feature_vals1.index.values, y=count_feature_vals1.values)
        count_feature_vals2 = train_df[train_df["totals.transactionRevenue"].astype("float32")>0][v].value_counts()
        trace2 = go.Bar(x=count_feature_vals2.index.values, y=count_feature_vals2.values)
        fig.append_trace(trace1, i+1, 1)
        fig.append_trace(trace2, i+1, 2)
    fig["layout"].update(height=height, 
                         width=width, 
                         title="Feature distribution: all data vs revenue > 0 data", 
                         showlegend=False, 
                         plot_bgcolor=plot_bgcolor, 
                         paper_bgcolor=paper_bgcolor)
    # 1. set subplot title font size
    # https://github.com/plotly/plotly.py/issues/985
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=font_size)
    # 2. set subplot tick font size
    for plot_num in range(1, (nrows * ncols) + 1):
        fig["layout"]["xaxis"+str(plot_num)].update(tickfont=dict(size=font_size,color="black"))
        fig["layout"]["yaxis"+str(plot_num)].update(tickfont=dict(size=font_size,color="black"), showgrid=False, showline=False, zeroline=False)
    # 3. log y axis 
    if log_yaxis:
        for row_num in range(0, nrows):
            if log_yaxis[row_num]:
                fig["layout"]["yaxis"+str(row_num * 2 + 1)].update(type="log")
                fig["layout"]["yaxis"+str(row_num * 2 + 2)].update(type="log")                
    iplot(fig)
feats_to_plot = ["channelGrouping"]
subplot_titles = generate_subplot_titles(feats_to_plot)
plot_feature_counts(feats_to_plot, subplot_titles, log_yaxis=[True])
train_df[["date", "visitStartTime"]].head(25)
train_df[["date", "visitStartTime"]].tail(25)
(train_df["visitStartTime.day"] - train_df["date.day"]).value_counts()
train_df[(train_df["visitStartTime.day"] - train_df["date.day"]) == -29 ][["visitStartTime.day", "date.day"]].head()
train_df[(train_df["visitStartTime.day"] - train_df["date.day"]) == -27 ][["visitStartTime.day", "date.day"]].head()
train_df[(train_df["visitStartTime.day"] - train_df["date.day"]) == -30 ][["visitStartTime.day", "date.day"]].head()
train_df = train_df.drop(["date", "date.year", "date.month", "date.day"], axis=1)
train_df.columns

feats_to_plot = ["device.browser",
                "device.deviceCategory",
                "device.isMobile",
                "device.operatingSystem"]
subplot_titles = generate_subplot_titles(feats_to_plot)
plot_feature_counts(feats_to_plot, subplot_titles, log_yaxis=[True, True, True, True])
feats_to_plot = ["geoNetwork.city",
                "geoNetwork.continent",
                "geoNetwork.country",
                "geoNetwork.metro",
                "geoNetwork.networkDomain",
                "geoNetwork.region",
                "geoNetwork.subContinent"]
subplot_titles = generate_subplot_titles(feats_to_plot)
plot_feature_counts(feats_to_plot, subplot_titles, log_yaxis=[True, True, True, True, True, True, True])
data_by_country_count = train_df["geoNetwork.country"].value_counts()
data = [dict(
    type = "choropleth",
    locations = data_by_country_count.index.values,
    locationmode = "country names",
    z = data_by_country_count.values,
    text = data_by_country_count.index.values,
    colorscale = "Viridis",
    autocolorscale = False,
    reversescale = True,
    marker = dict(line = dict(color = "rgb(180,180,180)", width = 0.5)),
    colorbar = dict(title="Count")
)]
layout = dict(title="Data distribution by country (all train data)", height=600, width=800, geo=dict(showframe=False))
fig = dict(data=data, layout=layout)
iplot(fig, validate=False)
data_by_country_count = train_df[train_df['totals.transactionRevenue'].astype("float32")>0]['geoNetwork.country'].value_counts()
data = [dict(
    type = "choropleth",
    locations = data_by_country_count.index.values,
    locationmode = "country names",
    z = data_by_country_count.values,
    text = data_by_country_count.index.values,
    colorscale = "Viridis",
    autocolorscale = False,
    reversescale = True,
    marker = dict(line = dict(color = "rgb(180,180,180)", width = 0.5)),
    colorbar = dict(title="Count")
)]
layout = dict(title="Data distribution by country (revenue > 0 data)", height=600, width=800, geo=dict(showframe=False))
fig = dict(data=data, layout=layout)
iplot(fig, validate=False)
feats_to_plot = ["totals.bounces",
                "totals.hits",
                "totals.newVisits",
                "totals.pageviews"]
subplot_titles = generate_subplot_titles(feats_to_plot)
plot_feature_counts(feats_to_plot, subplot_titles, log_yaxis=[True, True, True, True])
non_zero_revenue_train_df = train_df[train_df["totals.transactionRevenue"].astype("float32") > 0.0]
non_zero_revenue_train_df.shape