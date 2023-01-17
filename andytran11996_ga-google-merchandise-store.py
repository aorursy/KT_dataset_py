import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
%%time
df = pd.read_csv('../input/ga-allsessions-sorted.csv', dtype={'fullVisitorId': 'str'})
df.head(10)
numeric_cols = ["totals_hits", "totals_pageviews", "visitNumber", "visitStartTime", 'totals_bounces',  'totals_newVisits', 'totals_timeOnSite', 'totals_transactions', 'totals_transactionRevenue']    
for col in numeric_cols:
    df[col] = df[col].astype(float).fillna(0)
df["totals_transactionRevenue"] = df["totals_transactionRevenue"].astype('float') / 10**6
df['date'] = df['date'].astype(str)
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["month"]   = df['date'].dt.month
df["day"]     = df['date'].dt.day
df["weekday"] = df['date'].dt.weekday
gdf = df.groupby("fullVisitorId")["totals_transactionRevenue"].sum().reset_index()

notnull_visits = (df["totals_transactionRevenue"] > 0).sum()
notnull_customers = (gdf["totals_transactionRevenue"]>0).sum()
print("Non-zero revenue visits: %d, %.2f%%" %(notnull_visits,  notnull_visits*100 / df.shape[0]))
print("Non-zero revenue customers: %d, %.2f%%" %(notnull_customers,  notnull_customers*100 / gdf.shape[0]))

device_cols = ["device_browser", "device_deviceCategory", "device_operatingSystem"]
colors = ["#d6a5ff", "#fca6da", "#f4d39c", "#a9fcca"]
traces = []
for i, col in enumerate(device_cols):
    t = df[col].value_counts()
    traces.append(go.Bar(marker=dict(color=colors[i]),orientation="h", y = t.index[:15][::-1], x = t.values[:15][::-1]))

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Visits: Category", "Visits: Browser","Visits: OS"], print_grid=False)
fig.append_trace(traces[0], 1, 2)
fig.append_trace(traces[1], 1, 1)
fig.append_trace(traces[2], 1, 3)
fig['layout'].update(height=400, showlegend=False, title="Visits by Device Attributes")
iplot(fig)



fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Total Revenue: Category", "Total Revenue: Browser","Total Revenue: OS"], print_grid=False)
colors = ["red", "green", "purple"]
trs = []
for i, col in enumerate(device_cols):
    tmp = df.groupby(col).agg({"totals_transactionRevenue": "sum"}).reset_index().rename(columns={"totals_transactionRevenue" : "Total Revenue"})
    tmp = tmp.dropna().sort_values("Total Revenue", ascending = False)
    tr = go.Bar(x = tmp["Total Revenue"][::-1], orientation="h", marker=dict(opacity=0.5, color=colors[i]), y = tmp[col][::-1])
    trs.append(tr)

fig.append_trace(trs[0], 1, 2)
fig.append_trace(trs[1], 1, 1)
fig.append_trace(trs[2], 1, 3)
fig['layout'].update(height=400, showlegend=False, title="Total Revenue by Device Attributes")
iplot(fig)

device_cols = ["device_browser", "device_deviceCategory", "device_operatingSystem"]
fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Mean Revenue: Category", "Mean Revenue: Browser","Mean Revenue: OS"], print_grid=False)
colors = ["red", "green", "purple"]
trs = []
for i, col in enumerate(device_cols):
    tmp = df.groupby(col).agg({"totals_transactionRevenue": "mean"}).reset_index().rename(columns={"totals_transactionRevenue" : "Mean Revenue"})
    tmp = tmp.dropna().sort_values("Mean Revenue", ascending = False)
    tr = go.Bar(x = tmp["Mean Revenue"][::-1], orientation="h", marker=dict(opacity=0.5, color=colors[i]), y = tmp[col][::-1])
    trs.append(tr)

fig.append_trace(trs[0], 1, 2)
fig.append_trace(trs[1], 1, 1)
fig.append_trace(trs[2], 1, 3)
fig['layout'].update(height=400, showlegend=False, title="Mean Revenue by Device Attributes")
iplot(fig)

geo_cols = ['geoNetwork_continent','geoNetwork_subContinent']

colors = ["#d6a5ff", "#fca6da"]
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=["Visits : GeoNetwork Continent", "Visits : GeoNetwork subContinent"], print_grid=False)
trs = []
for i,col in enumerate(geo_cols):
    t = df[col].value_counts()
    tr = go.Bar(x = t.index[:20], marker=dict(color=colors[i]), y = t.values[:20])
    trs.append(tr)

fig.append_trace(trs[0], 1, 1)
fig.append_trace(trs[1], 1, 2)
fig['layout'].update(height=400, margin=dict(b=150), showlegend=False)
iplot(fig)


fig = tools.make_subplots(rows=1, cols=2, subplot_titles=["Total Revenue: Continent", "Total Revenue: SubContinent"], print_grid=False)

colors = ["blue", "orange"]
trs = []
for i, col in enumerate(geo_cols):
    tmp = df.groupby(col).agg({"totals_transactionRevenue": "sum"}).reset_index().rename(columns={"totals_transactionRevenue" : "Total Revenue"})
    tmp = tmp[tmp[col] != '(not set)']
    tmp = tmp.dropna().sort_values("Total Revenue", ascending = False)
    tr = go.Bar(y = tmp["Total Revenue"], orientation="v", marker=dict(opacity=0.5, color=colors[i]), x= tmp[col])
    trs.append(tr)

fig.append_trace(trs[0], 1, 1)
fig.append_trace(trs[1], 1, 2)
fig['layout'].update(height=450, margin=dict(b=200), showlegend=False)
iplot(fig)

geo_cols = ['geoNetwork_continent','geoNetwork_subContinent']
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=["Mean Revenue: Continent", "Mean Revenue: SubContinent"], print_grid=False)

colors = ["blue", "orange"]
trs = []
for i, col in enumerate(geo_cols):
    tmp = df.groupby(col).agg({"totals_transactionRevenue": "mean"}).reset_index().rename(columns={"totals_transactionRevenue" : "Mean Revenue"})
    tmp = tmp[tmp[col] != '(not set)']
    tmp = tmp.dropna().sort_values("Mean Revenue", ascending = False)
    tr = go.Bar(y = tmp["Mean Revenue"], orientation="v", marker=dict(opacity=0.5, color=colors[i]), x= tmp[col])
    trs.append(tr)

fig.append_trace(trs[0], 1, 1)
fig.append_trace(trs[1], 1, 2)
fig['layout'].update(height=450, margin=dict(b=200), showlegend=False)
iplot(fig)
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=["TrafficSource Campaign", "TrafficSource Medium"], print_grid=False)
tmp = df.copy()
colors = ["#d6a5ff", "#fca6da", "#f4d39c", "#a9fcca"]
t1 = tmp["trafficSource_campaign"].value_counts()
tmp = df.copy()
tmp = tmp[tmp.trafficSource_medium != '(not set)']
t2 = tmp["trafficSource_medium"].value_counts()
tr1 = go.Bar(x = t1.index, y = t1.values, marker=dict(color=colors[3]))
tr2 = go.Bar(x = t2.index, y = t2.values, marker=dict(color=colors[2]))
tr3 = go.Bar(x = t1.index[1:], y = t1.values[1:], marker=dict(color=colors[0]))
tr4 = go.Bar(x = t2.index[1:], y = t2.values[1:])

fig.append_trace(tr3, 1, 1)
fig.append_trace(tr2, 1, 2)
fig['layout'].update(height=400, margin=dict(b=100), showlegend=False)
iplot(fig)
tmp = df["channelGrouping"].value_counts()
tmp = tmp[:len(tmp)-1]
colors = ["#8d44fc", "#ed95d5", "#caadf7", "#6161b7", "#7e7eba", "#babad1"]
trace = go.Pie(labels=tmp.index, values=tmp.values, marker=dict(colors=colors))
layout = go.Layout(title="Channel Grouping", height=400)
fig = go.Figure(data = [trace], layout = layout)
iplot(fig, filename='basic_pie_chart')
df2 = df[df.trafficSource_medium == '(none)']
df2[df2['channelGrouping'] != 'Direct'].head()
vn = df["visitNumber"].value_counts()
def vn_bins(x):
    if x == 1:
        return "1" 
    elif x < 5:
        return "2-5"
    elif x < 10:
        return "5-10"
    elif x < 50:
        return "10-50"
    elif x < 100:
        return "50-100"
    else:
        return "100+"
    
vn = df["visitNumber"].apply(vn_bins).value_counts()

trace1 = go.Bar(y = vn.index[::-1], orientation="h" , x = vn.values[::-1], marker=dict(color="#7af9ad"))
layout = go.Layout(title="Visit Numbers Distribution", 
                   xaxis=dict(title="Frequency"),yaxis=dict(title="VisitNumber") ,
                   height=400, margin=dict(l=300, r=300))
figure = go.Figure(data = [trace1], layout = layout)
iplot(figure)

train = df.copy()
tmp = train['date'].value_counts().to_frame().reset_index().sort_values('index')
tmp = tmp.rename(columns = {"index" : "dateX", "date" : "visits"})

tr = go.Scattergl(mode="lines", x = tmp["dateX"].astype(str), y = tmp["visits"])
layout = go.Layout(title="Visits by date", height=400)
fig = go.Figure(data = [tr], layout = layout)
iplot(fig)

tmp = train.groupby("date").agg({"totals_transactionRevenue" : "sum"}).reset_index()
tmp = tmp.rename(columns = {"date" : "dateX", "totals_transactionRevenue" : "mean_revenue"})
tr = go.Scattergl(mode="lines", x = tmp["dateX"].astype(str), y = tmp["mean_revenue"])
layout = go.Layout(title="Total Revenue by date", height=400)
fig = go.Figure(data = [tr], layout = layout)
iplot(fig)

tmp = train.groupby("date").agg({"totals_transactionRevenue" : "mean"}).reset_index()
tmp = tmp.rename(columns = {"date" : "dateX", "totals_transactionRevenue" : "mean_revenue"})
tr = go.Scattergl(mode="lines", x = tmp["dateX"].astype(str), y = tmp["mean_revenue"])
layout = go.Layout(title="Mean Revenue by date", height=400)
fig = go.Figure(data = [tr], layout = layout)
iplot(fig)
fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Visits by Month", "Visits by MonthDay", "Visits by WeekDay"], print_grid=False)
trs = []
for i,col in enumerate(["month", "day", "weekday"]):
    t = df[col].value_counts()
    tr = go.Bar(x = t.index, marker=dict(color=colors[i]), y = t.values)
    trs.append(tr)

fig.append_trace(trs[0], 1, 1)
fig.append_trace(trs[1], 1, 2)
fig.append_trace(trs[2], 1, 3)
fig['layout'].update(height=400, showlegend=False)
iplot(fig)


tmp1 =df.groupby('month').agg({"totals_transactionRevenue" : "sum"}).reset_index()
tmp2 = df.groupby('day').agg({"totals_transactionRevenue" : "sum"}).reset_index()
tmp3 = df.groupby('weekday').agg({"totals_transactionRevenue" : "sum"}).reset_index()

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["Total Revenue by Month", "Total Revenue by MonthDay", "Total Revenue by WeekDay"], print_grid=False)
tr1 = go.Bar(x = tmp1.month, marker=dict(color="red", opacity=0.5), y = tmp1['totals_transactionRevenue'].values)
tr2 = go.Bar(x = tmp2.day, marker=dict(color="orange", opacity=0.5), y = tmp2['totals_transactionRevenue'].values)
tr3 = go.Bar(x = tmp3.weekday, marker=dict(color="green", opacity=0.5), y = tmp3['totals_transactionRevenue'].values)

fig.append_trace(tr1, 1, 1)
fig.append_trace(tr2, 1, 2)
fig.append_trace(tr3, 1, 3)
fig['layout'].update(height=400, showlegend=False)
iplot(fig)


tmp1 =df.groupby('month').agg({"totals_transactionRevenue" : "mean"}).reset_index()
tmp2 = df.groupby('day').agg({"totals_transactionRevenue" : "mean"}).reset_index()
tmp3 = df.groupby('weekday').agg({"totals_transactionRevenue" : "mean"}).reset_index()

fig = tools.make_subplots(rows=1, cols=3, subplot_titles=["MeanRevenue by Month", "MeanRevenue by MonthDay", "MeanRevenue by WeekDay"], print_grid=False)
tr1 = go.Bar(x = tmp1.month, marker=dict(color="red", opacity=0.5), y = tmp1['totals_transactionRevenue'].values)
tr2 = go.Bar(x = tmp2.day, marker=dict(color="orange", opacity=0.5), y = tmp2['totals_transactionRevenue'].values)
tr3 = go.Bar(x = tmp3.weekday, marker=dict(color="green", opacity=0.5), y = tmp3['totals_transactionRevenue'].values)

fig.append_trace(tr1, 1, 1)
fig.append_trace(tr2, 1, 2)
fig.append_trace(tr3, 1, 3)
fig['layout'].update(height=400, showlegend=False)
iplot(fig)
plt.figure(figsize=(12,6))
time_on_site = df['totals_timeOnSite'].dropna().astype('int')
sns.distplot(time_on_site)
plt.title("Log Distribution of Time on site");

u_behavior_cols = ['totals_pageviews', 'totals_timeOnSite', 'totals_hits']

from sklearn.preprocessing import MinMaxScaler
#pca_x_ = train_df[u_behavior_cols].fillna(0).values
df_gid = df.groupby("fullVisitorId").agg(['sum'])[u_behavior_cols].fillna(0).values
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(df_gid)
#train_df.groupby("fullVisitorId").agg({'totals.transactions':'sum', 'totals.pageviews':'sum'})[u_behavior_cols].fillna(0).max()
from sklearn.decomposition import PCA
pca = PCA(n_components=2) 
pca.fit(df_gid)
pca_x2 = pca.transform(df_gid)
trace1 = go.Scattergl(
    x=pca_x2[:,0],
    y=pca_x2[:,1],
    mode='markers',
    marker=dict(
        size=2,
        opacity=0.8
    )
)
iplot([trace1])
from sklearn.cluster import MeanShift, AgglomerativeClustering, MiniBatchKMeans
clustering = MiniBatchKMeans(n_clusters=3)
classes = clustering.fit_predict(scaled_x)

import colorlover as cl

class_colors = ['#4285f3','#34a853','#fbbc05','#ea4335']
colors = [class_colors[c] for c in clustering.labels_]
trace = go.Scattergl(
    x=pca_x2[:,0],
    y=pca_x2[:,1],
    mode='markers',
    marker=dict(
        size=4,
        color=colors,
        opacity=1
    )
)
iplot([trace])

agg_dict = {
    'totals.transactionRevenue'
}
cluster_df = df.groupby("fullVisitorId").mean().reset_index()[['totals_transactionRevenue', 'totals_hits', 'totals_pageviews', 'totals_timeOnSite', 'totals_bounces', 'visitNumber']]
cluster_df2 = df.groupby("fullVisitorId").mean()[['totals_transactionRevenue', 'totals_hits', 'totals_pageviews', 'totals_timeOnSite', 'totals_bounces', 'visitNumber']]

for c in range(3):
    cluster_df.loc[clustering.labels_ == c, 'cluster'] = str(c+1)
    cluster_df2.loc[clustering.labels_ == c, 'cluster'] = str(c+1)
df['cluster'] = df['fullVisitorId'].apply(lambda idx: cluster_df2.loc[idx].cluster)
df.head()
cluster_df.head(10)
cluster_df2_sum = cluster_df.groupby(['cluster']).sum().reset_index()
cluster_df2_mean = cluster_df.groupby(['cluster']).mean().reset_index()
cluster_df2_mean
cols = [ 'totals_hits', 'totals_timeOnSite', 'totals_pageviews','totals_transactionRevenue', 'visitNumber', 'totals_bounces']
names = ['Hits', 'Time on site', 'Page views', 'Revenue', 'Visit number', 'Bounce rate']
cluster_names = ['C1', 'C2', 'C3']
data = []

fig = tools.make_subplots(rows=1, cols=len(cols), subplot_titles=tuple(names))
for i in range(len(cols)):
    trace = go.Bar(
        x= cluster_names,
        y= cluster_df2_mean[cols[i]].values
    )
    fig.append_trace(trace, 1, i+1)
iplot(fig, filename='grouped-bar')
fig = tools.make_subplots(rows=1, cols=3, subplot_titles=tuple(cluster_names))
fig['layout'].update(title='TrafficSource Medium')
for c in range(1,4):
    ts_c1 = pd.crosstab(index=df.loc[df['cluster'] == str(c), 'trafficSource_medium'],  # Make a crosstab
                        columns="count")               # Name the count column
    if '(not set)' in ts_c1.index:
        ts_c1 = ts_c1.drop(['(not set)'])
    ts_c1 = ts_c1.sort_values(['count'], ascending=False)
    trace = go.Bar(x = ts_c1.index, y = ts_c1.values[:,0])
    fig.append_trace(trace, 1, c)

iplot(fig, filename='grouped-bar')


