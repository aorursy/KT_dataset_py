import numpy as np 
import pandas as pd 
import json
from pandas.io.json import json_normalize
import datetime as dt
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import os
IS_LOCAL=False
if(IS_LOCAL):
    PATH="../google-analytics-customer-revenue-prediction/input/"    
else:
    PATH="../input/"
print(os.listdir(PATH))
onerow = pd.read_csv(PATH+'train.csv',nrows=1)
pd.concat([onerow.T, onerow.dtypes.T], axis=1, keys=['Example', 'Type'])
#the columns that will be parsed to extract the fields from the jsons
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']

def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    return data_df
%%time
train_df = read_parse_dataframe('train.csv')
print("Train set:",train_df.shape[0]," rows, ", train_df.shape[1],"columns")
train_df.head()
print(dt.datetime.fromtimestamp(train_df['visitId'][0]).isoformat())
def process_date_time(data_df):
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    return data_df
train_df = process_date_time(train_df)
print("Train set:",train_df.shape[0]," rows, ", train_df.shape[1],"columns")
%%time
test_df = read_parse_dataframe('test.csv')
test_df = process_date_time(test_df)
print("Test set:",test_df.shape[0]," rows, ", test_df.shape[1],"columns")
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return df.loc[~(df['Total']==0)]
missing_data(train_df)
missing_data(test_df)
def get_categories(data, val):
    tmp = data[val].value_counts()
    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()
def draw_trace_bar(data, title, xlab, ylab,color='Blue'):
    trace = go.Bar(
            x = data.head(30)['index'],
            y = data.head(30)['Number'],
            marker=dict(color=color),
            text=data.head(30)['index']
        )
    data = [trace]

    layout = dict(title = title,
              xaxis = dict(title = xlab, showticklabels=True, tickangle=15,
                          tickfont=dict(
                            size=9,
                            color='black'),), 
              yaxis = dict(title = ylab),
              hovermode = 'closest'
             )
    fig = dict(data = data, layout = layout)
    iplot(fig, filename='draw_trace')
draw_trace_bar(get_categories(train_df,'channelGrouping'), "Channel grouping", "Channel grouping", "Number", "Lightblue")
def get_feature_distribution(data, feature):
    # Get the count for each label
    label_counts = data[feature].value_counts()
    # Get total number of samples
    total_samples = len(data)
    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        count = label_counts.values[i]
        percent = int((count / total_samples) * 10000)/100
        print("{:<30s}:   {} or {}%".format(label, count, percent))

get_feature_distribution(train_df,'channelGrouping')
get_feature_distribution(train_df,'socialEngagementType')
device_cols = train_df.columns[train_df.columns.str.contains('device')].T.tolist()
print("There are ",len(device_cols),"columns with device attributes:\n",device_cols)
const_device_cols = []
for i, col in enumerate(device_cols):
    if(len(train_df[col].value_counts())==1):
        const_device_cols.append(col)
print("There are ",len(const_device_cols),"columns with unique value for device attributes:\n",const_device_cols)
def show_features(data,features):
    color = ["red", "blue", "green", "magenta", "yellow", "lightblue", "gold", "tomato", "grey",
            "lightgreen", "red", "blue", "green", "magenta", "yellow", "brown", "grey", "tomato"]
    for i,feature in enumerate(features):
        draw_trace_bar(get_categories(train_df,feature), 
                    feature, feature, "Number", color[i])
var_cols = [item for item in device_cols if item not in const_device_cols]
show_features(train_df,var_cols)
def plot_heatmap_count(data_df, feature1, feature2, feature3='channelGrouping', color="Greens", title="", height=16, width=16):
    tmp = data_df.groupby([feature1, feature2])[feature3].count()
    df1 = tmp.reset_index()
    matrix = df1.pivot(feature1, feature2, feature3)
    fig, (ax1) = plt.subplots(ncols=1, figsize=(width,height))
    sns.heatmap(matrix, 
        xticklabels=matrix.columns,
        yticklabels=matrix.index,ax=ax1,linewidths=.1,linecolor='black',annot=True,cmap=color)
    plt.title(title, fontsize=14)
    plt.show()
    
def plot_heatmap_sum(data_df, feature1, feature2, feature3='channelGrouping', color="Greens", title="", height=16, width=16):
    tmp = data_df.groupby([feature1, feature2])[feature3].sum()
    df1 = tmp.reset_index()
    matrix = df1.pivot(feature1, feature2, feature3)
    fig, (ax1) = plt.subplots(ncols=1, figsize=(width,height))
    sns.heatmap(matrix, 
        xticklabels=matrix.columns,
        yticklabels=matrix.index,ax=ax1,linewidths=.1,linecolor='black',annot=True,cmap=color)
    plt.title(title, fontsize=14)
    plt.show()
plot_heatmap_count(train_df, 'device_browser', 'device_operatingSystem',color='Reds',title="Device Browsers vs. Device OS")
plot_heatmap_count(train_df, 'device_browser','device_deviceCategory', color='Blues',title="Device Browser vs. Device Category", height=12, width=8)
plot_heatmap_count(train_df, 'device_deviceCategory', 'device_isMobile', color='viridis',title="Device is mobile vs. Device Category", width=6, height=4)
geo_cols = train_df.columns[train_df.columns.str.contains('geoNetwork')].T.tolist()
print("There are ",len(geo_cols),"columns with geoNetwork attributes:\n",geo_cols)
const_geo_cols = []
for i, col in enumerate(geo_cols):
    if(len(train_df[col].value_counts())==1):
        const_geo_cols.append(col)
print("There are ",len(const_geo_cols),"columns with unique value for geoNetwork attributes:\n",const_geo_cols)
var_cols = [item for item in geo_cols if item not in const_geo_cols]
show_features(train_df,var_cols)
tmp = train_df['geoNetwork_country'].value_counts()
country_visits = pd.DataFrame(data={'geoNetwork_country': tmp.values}, index=tmp.index).reset_index()
country_visits.columns = ['Country', 'Visits']
def plot_country_map(data, location, z, legend, title, colormap='Rainbow'):
    data = dict(type = 'choropleth', 
                colorscale = colormap,
                autocolorscale = False,
                reversescale = False,
               locations = data[location],
               locationmode = 'country names',
               z = data[z], 
               text = data[z],
               colorbar = {'title':legend})
    layout = dict(title = title, 
                 geo = dict(showframe = False, 
                         projection = {'type': 'natural earth'}))
    choromap = go.Figure(data = [data], layout=layout)
    iplot(choromap)
plot_country_map(country_visits, 'Country', 'Visits', 'Visits', 'Visits per country')
tot_cols = train_df.columns[train_df.columns.str.contains('totals')].T.tolist()
print("There are ",len(tot_cols),"columns with Totals attributes:\n",tot_cols)
const_tot_cols = []
for i, col in enumerate(tot_cols):
    if(len(train_df[col].value_counts())==1):
        const_tot_cols.append(col)
print("There are ",len(const_tot_cols),"columns with unique value for Totals attributes:\n",const_tot_cols)
var_cols = [item for item in tot_cols if item not in const_tot_cols]
show_features(train_df,var_cols,12,4)
train_df['totals_transactionRevenue'] = pd.to_numeric(train_df['totals_transactionRevenue'])
df = train_df[train_df['totals_transactionRevenue'] > 0]['totals_transactionRevenue']
f, ax = plt.subplots(1,1, figsize=(16,4))
plt.title("Distribution of totals: transaction revenue")
sns.kdeplot(df, color="green")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.ylabel('Density plot', fontsize=12)
plt.xlabel('Transaction revenue', fontsize=12)
locs, labels = plt.xticks()
plt.show()
plt.figure(figsize=(12,6))
sns.distplot(np.log1p(df),color="darkgreen",bins=50)
plt.xlabel("Log(total transaction revenue)");
plt.title("Logarithmic distribution of total transaction revenue (non-zeros)");
# select the visits with non-zero transaction revenue
non_zero = train_df[train_df['totals_transactionRevenue']>0]
tmp = non_zero['geoNetwork_country'].value_counts()
country_visits = pd.DataFrame(data={'geoNetwork_country': tmp.values}, index=tmp.index).reset_index()
country_visits.columns = ['Country', 'Visits']
plot_country_map(country_visits, 'Country', 'Visits', 'Visits', 'Visits with non zero transactions')
# select the visits with non-zero transaction revenue and calculate the sums
non_zero = train_df[train_df['totals_transactionRevenue']>0]
tmp = non_zero.groupby(['geoNetwork_country'])['totals_transactionRevenue'].sum()
country_total = pd.DataFrame(data={'total': tmp.values}, index=tmp.index).reset_index()
country_total.columns = ['Country', 'Total']
country_total['Total']  = np.log1p(country_total['Total'])
plot_country_map(country_total, 'Country', 'Total', 'Total(log)', 'Total revenues per country (log scale)')
non_zero[['fullVisitorId','visitNumber', 'totals_transactionRevenue', 'channelGrouping']].sort_values(['totals_transactionRevenue', 'fullVisitorId'], ascending=[0,0]).head(10)
non_zero[['fullVisitorId','visitNumber', 'totals_transactionRevenue', 'channelGrouping']].sort_values(['visitNumber', 'totals_transactionRevenue'], ascending=[0,0]).head(10)
# select the visits with non-zero transaction revenue and calculate the sums
non_zero = train_df[train_df['totals_transactionRevenue']>0]
tmp = non_zero.groupby(['channelGrouping', 'geoNetwork_subContinent'])['totals_transactionRevenue'].sum()
channel_total = pd.DataFrame(data={'total': tmp.values}, index=tmp.index).reset_index()
channel_total.columns = ['Channel', 'Subcontinent', 'Total']
plot_heatmap_sum(non_zero, 'geoNetwork_subContinent','channelGrouping',  'totals_transactionRevenue','rainbow',"Total transactions per channel and subcontinent", width=16, height=6)
ts_cols = train_df.columns[train_df.columns.str.contains('trafficSource')].T.tolist()
print("There are ",len(ts_cols),"columns with Totals attributes:\n",ts_cols)
const_ts_cols = []
for i, col in enumerate(ts_cols):
    if(len(train_df[col].value_counts())==1):
        const_ts_cols.append(col)
print("There are ",len(const_ts_cols),"columns with unique value for Traffic Source attributes:\n",const_ts_cols)
var_cols = [item for item in ts_cols if item not in const_ts_cols]
show_features(train_df,var_cols)
var_cols = ['year','month','day','weekday']
show_features(train_df,var_cols)
def plot_scatter_data(data, xtitle, ytitle, title, color='blue'):
    trace = go.Scatter(
        x = data.index,
        y = data.values,
        name=ytitle,
        marker=dict(
            color=color,
        ),
        mode='lines+markers'
    )
    data = [trace]
    layout = dict(title = title,
              xaxis = dict(title = xtitle), yaxis = dict(title = ytitle),
             )
    fig = dict(data=data, layout=layout)
    iplot(fig, filename='lines')
count_all = train_df.groupby('date')['totals_transactionRevenue'].agg(['size'])
count_all.columns = ["Total"]
count_all = count_all.sort_index()
plot_scatter_data(count_all['Total'],'Date', 'Total','Total count of visits (including zero transactions)','green')
count_nonzero = train_df.groupby('date')['totals_transactionRevenue'].agg(['count'])
count_nonzero.columns = ["Total"]
count_nonzero = count_nonzero.sort_index()
plot_scatter_data(count_nonzero['Total'],'Date', 'Total','Total non-zero transaction visits','darkblue')
total_nonzero = train_df.groupby('date')['totals_transactionRevenue'].agg(['sum'])
total_nonzero.columns = ["Total"]
total_nonzero = total_nonzero.sort_index()
plot_scatter_data(total_nonzero['Total'],'Date', 'Total','Total non-zero transaction amounts','red')
channels = list(train_df['channelGrouping'].unique())
data = []
for channel in channels:
    subset = train_df[train_df['channelGrouping']==channel]
    subset = subset.groupby('date')['totals_transactionRevenue'].agg(['sum'])
    subset.columns = ["Total"]
    subset = subset.sort_index()
    trace = go.Scatter(
        x = subset['Total'].index,
        y = subset['Total'].values,
        name=channel,
        mode='lines'
    )
    data.append(trace)
layout= go.Layout(
    title= 'Total amount of non-zero transactions per day, grouped by channel',
    xaxis = dict(title = 'Date'), yaxis = dict(title = 'Total'),
    showlegend=True,
)
fig = dict(data=data, layout=layout)
iplot(fig, filename='lines')
opsys = list(train_df['device_operatingSystem'].unique())
data = []
for os in opsys:
    subset = train_df[train_df['device_operatingSystem']==os]
    subset = subset.groupby('date')['totals_transactionRevenue'].agg(['sum'])
    subset.columns = ["Total"]
    subset = subset.sort_index()
    trace = go.Scatter(
        x = subset['Total'].index,
        y = subset['Total'].values,
        name=os,
        mode='lines'
    )
    data.append(trace)
layout= go.Layout(
    title= 'Total amount of non-zero transactions per day, grouped by OS',
    xaxis = dict(title = 'Date'), yaxis = dict(title = 'Total'),
    showlegend=True,
)
fig = dict(data=data, layout=layout)
iplot(fig, filename='lines')
total_test = test_df.groupby('date')['fullVisitorId'].agg(['count'])
total_test.columns = ["Total"]
total_test = total_test.sort_index()
plot_scatter_data(total_test['Total'],'Date', 'Total','Total count of visits per day (test set)','magenta')