import gc
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from scipy.stats import norm
import json
import datetime
# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)

# 'train.csv', 'sample_submission.csv', 'test.csv'
def load_df(file_name = 'train.csv', nrows = None):
    """ Read csv and convert json columns. Author: Julián Peller. """
    
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv('../input/{}'.format(file_name),
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
train, test = load_df('train.csv'), load_df('test.csv')
train.head()
values_name = ['Train rows (visits)', 'Test rows (visits)', 'Train unique users', 'Test unique users']
values = [len(train), len(test), train['fullVisitorId'].nunique(), test['fullVisitorId'].nunique()]
plt.figure(figsize=(8,4))
plt.title("Basic statistics")
ax = sns.barplot(x=values_name, y=values, palette='Blues_d')
non_missing = len(train[~train['totals_transactionRevenue'].isnull()])
num_visitors = train[~train['totals_transactionRevenue'].isnull()]['fullVisitorId'].nunique()
print("totals_transactionRevenue has {} non-missing values or {:.3f}% (train set)"
      .format(non_missing, 100*non_missing/len(train)))
print("Only {} unique users have transactions or {:.3f}% (train set)"
      .format(num_visitors, num_visitors/train['fullVisitorId'].nunique()))
# Logn Distplot
revenue = train['totals_transactionRevenue'].dropna().astype('float64')
plt.figure(figsize=(10,4))
plt.title("Natural log Distribution - Transactions revenue")
ax1 = sns.distplot(np.log(revenue), color="#006633", fit=norm)
# Log10 Distplot
plt.figure(figsize=(10,4))
plt.title("Log10 Distribution - Transactions revenue")
ax1 = sns.distplot(np.log10(revenue), color="#006633", fit=norm)

# Fill missing with 0 and convert to numerical
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0).astype('int64')
g = train[train['totals_transactionRevenue'] > 0][['fullVisitorId', 'totals_transactionRevenue']]
sum_transactions = g.groupby('fullVisitorId')['totals_transactionRevenue'].sum()
plt.figure(figsize=(10,4))
plt.title("Log10 Distribution - Sum of transactions per user")
ax1 = sns.distplot(np.log10(sum_transactions), color="#005c99", fit=norm)
def convert_to_datetime(frame):
    frame['date'] = frame['date'].astype(str)
    frame['date'] = frame['date'].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    frame['date'] = pd.to_datetime(frame['date'])
    return frame

train = convert_to_datetime(train)
test = convert_to_datetime(test)
# Visits by time train
tmp_train = train['date'].value_counts().to_frame().reset_index().sort_values('index')
tmp_train = tmp_train.rename(columns = {"date" : "visits"}).rename(columns = {"index" : "date"})
# Visits by time test
tmp_test = test['date'].value_counts().to_frame().reset_index().sort_values('index')
tmp_test = tmp_test.rename(columns = {"date" : "visits"}).rename(columns = {"index" : "date"})
# Plot visits
trace1 = go.Scatter(x=tmp_train.date.astype(str), y=tmp_train.visits,
                    opacity = 0.8, line = dict(color = '#ff751a'), name= 'Train')
trace2 = go.Scatter(x=tmp_test.date.astype(str), y=tmp_test.visits,
                    opacity = 0.8, line = dict(color = '#75a3a3'), name= 'Test')
traces = [trace1, trace2]

layout = dict(
    title= "Visits by date",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)

fig = dict(data= traces, layout=layout)
iplot(fig)


# Revenue by time
train_date_sum = train.groupby('date')['totals_transactionRevenue'].sum().to_frame().reset_index()
# Plot
trace_date = go.Scatter(x=train_date_sum.date.astype(str), 
                        y=train_date_sum['totals_transactionRevenue'].apply(lambda x: np.log(x)), opacity = 0.8)
layout = dict(
    title= "Log Revenue by date",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)

fig = dict(data= [trace_date], layout=layout)
iplot(fig)
def missing_plot(frame, set_name, palette):
    nan_ratio = frame.isna().sum()/len(frame)
    nan_ratio = nan_ratio.to_frame().reset_index().rename({'index': 'column', 0: 'missing_percentage'},axis=1)
    nan_ratio.sort_values(by=['missing_percentage'], ascending=False, inplace=True)
    plt.figure(figsize=(8,6))
    plt.title("Columns with missing values - {}".format(set_name))
    ax = sns.barplot(x='missing_percentage', y='column', orient='h',
                     data=nan_ratio[nan_ratio['missing_percentage'] > 0],
                     palette= palette)

missing_plot(train, 'train', 'Blues_d')
missing_plot(train, 'test', 'Greens_d')
train_const_cols = [c for c in train.columns if len(train[c].unique()) == 1]
test_const_cols = [c for c in test.columns if len(test[c].unique()) == 1]
print("{} columns with a unique value on train set".format(len(train_const_cols)))
print("{} columns with a unique value on test set".format(len(test_const_cols)))
print("Same columns for train and test set: ", train_const_cols == test_const_cols)
train.drop(train_const_cols, axis=1, inplace=True)
test.drop(test_const_cols, axis=1, inplace=True)
print("Shape after dropping: train {}, test {}".format(train.shape, test.shape))
# Flag visits with revenue
train['has_revenue'] = train['totals_transactionRevenue'].apply(lambda x: 1 if x > 0 else 0)
def barplot_percentage(count_feat, color1= 'rgb(55, 83, 109)', 
                       color2= 'rgb(26, 118, 255)',num_bars= None):

    train_channel = 100*train[count_feat].value_counts()/len(train)
    train_channel = train_channel.to_frame().reset_index()
    test_channel = 100*test[count_feat].value_counts()/len(test)
    test_channel = test_channel.to_frame().reset_index()
    if num_bars:
        train_channel = train_channel.head(num_bars)
        test_channel = test_channel.head(num_bars)

    trace0 = go.Bar(
        x=train_channel['index'],
        y=train_channel[count_feat],
        name='Train set',
        marker=dict(color=color1)
    )
    trace1 = go.Bar(
        x=test_channel['index'],
        y=test_channel[count_feat],
        name='Test set',
        marker=dict(color=color2,)
    )

    layout = go.Layout(
        title='{} grouping'.format(count_feat),
        xaxis=dict(
            tickfont=dict(size=14, color='rgb(107, 107, 107)')
        ),
        yaxis=dict(
            title='Percentage of visits',
            titlefont=dict(size=16, color='rgb(107, 107, 107)'),
            tickfont=dict(size=14, color='rgb(107, 107, 107)')
        ),
        legend=dict(
            x=1.0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )

    fig = go.Figure(data=[trace0, trace1], layout=layout)
    iplot(fig)
barplot_percentage('channelGrouping')
channel_order = ['Referral', 'Display', 'Paid Search', 'Direct', 'Organic Search', '(Other)', 'Social', 'Affiliates']
plt.figure(figsize=(10,4))
plt.title("Percentage of visits with revenue per channel")
sns.barplot(x='channelGrouping', y='has_revenue', data=train, order=channel_order, palette='Greens_d')
revenue_channel = train.groupby('channelGrouping')['totals_transactionRevenue'].sum()
revenue_channel = revenue_channel.to_frame().reset_index()
plt.figure(figsize=(10,4))
plt.title("Mean revenue for each channel")
ax = sns.barplot(x='channelGrouping', y='totals_transactionRevenue', data=revenue_channel, order=channel_order, palette='Greens_d')
plt.figure(figsize=(10,4))
plt.title("Visit number distribution")
ax1 = sns.kdeplot(train['visitNumber'], label='Train set', color="#005c99")
ax2 = sns.kdeplot(test['visitNumber'], label='Test set', color="#e68a00")
plt.figure(figsize=(10,4))
plt.title("Number of visits and revenue")
ax = sns.scatterplot(x='visitNumber', y='totals_transactionRevenue',
                     data=train,color='orange', hue='has_revenue') #[train['has_revenue'] > 0]
def plotmap(frame, z_var, countries_col, title, colorscale, rcolor=True):

    data = [ dict(
            type = 'choropleth',
            autocolorscale = False,
            colorscale = colorscale,
            showscale = True,
            reversescale = rcolor,
            locations = frame[countries_col],
            z = frame[z_var],
            locationmode = 'country names',
            text = frame[countries_col],
            marker = dict(line = dict(color = '#fff', width = 2))
        )           
    ]

    layout = dict(
        height=680,
        #width=1200,
        title = title,
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(type = 'mercator'),
        ),
    )
    fig = dict(data=data, layout=layout)
    iplot(fig)

colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 
              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 
              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 
              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
    
# Plot world map - total visits
tmp = train["geoNetwork_country"].value_counts().to_frame().reset_index()
plotmap(tmp, 'geoNetwork_country', 'index', 'Total visits by Country', colorscale, False)

colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
        [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]
# Plot world map - mean revenue
tmp = train.groupby("geoNetwork_country").agg({"totals_transactionRevenue" : "mean"}).reset_index()
plotmap(tmp, 'totals_transactionRevenue','geoNetwork_country', 'Mean revenue by country', colorscale)
# Plot world map - total revenue
tmp = train.groupby("geoNetwork_country").agg({"totals_transactionRevenue" : "sum"}).reset_index()
plotmap(tmp, 'totals_transactionRevenue','geoNetwork_country', 'Total revenue by country', colorscale)
tmp1 = train["geoNetwork_continent"].value_counts().to_frame().reset_index()
tmp2 = train.groupby("geoNetwork_continent")["totals_transactionRevenue"].sum().to_frame().reset_index()
fig = {
  "data": [
    {
        "values": tmp1['geoNetwork_continent'],
        "labels": tmp1['index'],
        "name": "Visits",
        "domain": {"x": [0, 0.46]},
        "hoverinfo":"label+percent+name",
        "hole": .5,
        "type": "pie",
        #"textinfo": "none"
    },
    {
        "values": tmp2['totals_transactionRevenue'],
        "labels": tmp2['geoNetwork_continent'],
        "name": "Revenue",
        #"textposition":"inside",
        "domain": {"x": [.54, 1]},
        "hoverinfo":"label+percent+name",
        "hole": .5,
        "type": "pie",
        #"textinfo": "none"
    }],
  "layout": {
        "title":"Visits and Revenue by Continent",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Visits",
                "x": 0.18,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Revenue",
                "x": 0.85,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
barplot_percentage('geoNetwork_networkDomain', num_bars= 10, 
                   color1='rgb(38, 115, 77)', color2='rgb(102, 204, 153)')
bounces_nan = train[train['totals_bounces'].isnull()]['totals_transactionRevenue'].sum()
bounces_1 = train[~train['totals_bounces'].isnull()]['totals_transactionRevenue'].sum()
print("Revenue for bounce missing: {}, revenue for bounce '1': {}".format(bounces_nan, bounces_1))
plt.figure(figsize=(8,4))
plt.title("Bounces count")
ax = sns.countplot(x='totals_bounces', data=train.fillna('nan'))
plt.figure(figsize=(8,4))
plt.title("New visits count")
ax = sns.countplot(x='totals_newVisits', data=train.fillna('nan'))
plt.figure(figsize=(10,4))
plt.title("Hits distribution")
ax1 = sns.kdeplot(train['totals_hits'].astype('float64'), color="#006633", shade=True)
plt.figure(figsize=(10,4))
plt.title("Page views distribution")
ax2 = sns.kdeplot(train[train['has_revenue'] == 0]['totals_hits'].astype('float64'),
                  label='No revenue', color="#0000ff")
ax2 = sns.kdeplot(train[train['has_revenue'] == 1]['totals_hits'].astype('float64'),
                  label='Has revenue', color="#ff6600")
plt.figure(figsize=(10,4))
plt.title("Hits vs Log1p Revenue")
ax = sns.scatterplot(x=train['totals_hits'].astype('float64'), y=np.log1p(train['totals_transactionRevenue']))
plt.figure(figsize=(10,4))
plt.title("Page views distribution")
ax3 = sns.kdeplot(train['totals_pageviews'].dropna().astype('float64'), color="#006633", shade=True)
plt.figure(figsize=(10,4))
plt.title("Page views distribution")
ax4 = sns.kdeplot(train[train['has_revenue'] == 0]['totals_pageviews'].dropna().astype('float64'),
                  label='No revenue', color="#0000ff")
ax4 = sns.kdeplot(train[train['has_revenue'] == 1]['totals_pageviews'].dropna().astype('float64'),
                  label='Has revenue', color="#ff6600")
plt.figure(figsize=(10,4))
plt.title("Page views vs Log1p Revenue")
ax = sns.scatterplot(x=train['totals_pageviews'].dropna().astype('float64'), y=np.log1p(train['totals_transactionRevenue']))
plt.figure(figsize=(10,6))
plt.title("Hits vs Page views")
tmp = train[['totals_hits', 'totals_pageviews', 'has_revenue']].copy()
tmp['totals_hits'] = tmp['totals_hits'].astype('float64')
tmp['totals_pageviews'] = tmp['totals_pageviews'].dropna().astype('float64')
ax = sns.scatterplot(x='totals_hits', y='totals_pageviews', hue='has_revenue', data=tmp)
# Group and plot revenue
def group_revenue(group_col, title, sum_values=True, palette='Blues_d', size=(8,5)):
    if sum_values:
        tmp = train.groupby(group_col)['totals_transactionRevenue'].sum()
    else:
        tmp = train.groupby(group_col)['totals_transactionRevenue'].mean()
    tmp = tmp.to_frame().reset_index().sort_values('totals_transactionRevenue', ascending=False)
    tmp = tmp[tmp['totals_transactionRevenue'] > 0]
    plt.figure(figsize=size)
    plt.title(title)
    ax = sns.barplot(y=tmp[group_col], x= tmp.totals_transactionRevenue, orient='h', palette=palette)
# Visits
barplot_percentage('device_browser', num_bars= 7)
# Revenue
group_revenue('device_browser', 'Total revenue by browser', True, size=(9,5))
group_revenue('device_browser', 'Mean revenue by browser', False, 'Greens_d',size=(9,5))
# Pie chart
colors = ['#5c8a8a', '#94b8b8', '#b3cccc']
dev_train = train['device_deviceCategory'].value_counts().to_frame().reset_index()
dev_test = test['device_deviceCategory'].value_counts().to_frame().reset_index()
trace1 = go.Pie(labels=dev_train['index'], values=dev_train.device_deviceCategory,
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=dev_test['index'], values=dev_test.device_deviceCategory,
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))
layout = dict(title= "Device category - train and test", height=400)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)
# Revenue
group_revenue('device_deviceCategory', 'Mean revenue by device category', False, 'Greens_d', size=(8, 3.6))
barplot_percentage('device_operatingSystem', num_bars = 7,
                   color1='rgb(204, 82, 0)', color2='rgb(255, 163, 102)')
# Revenue
group_revenue('device_operatingSystem', 'Total revenue by OS', True, 'Blues_d', size=(9, 4))
group_revenue('device_operatingSystem', 'Mean revenue by OS', False, 'Greens_d', size=(9, 4))
tmp = train.groupby('fullVisitorId').size().value_counts().to_frame().reset_index()
sum_ = tmp[tmp['index'].astype('int16') > 5][0].sum()
tmp = tmp.head(5).append(pd.DataFrame({'index': ['more than 5'], 0: [sum_]})).reset_index()
plt.figure(figsize=(10,4))
plt.title("Visits per user")
ax = sns.barplot(x=tmp['index'], y=tmp[0], palette='Blues_d')
# Plot from https://www.kaggle.com/shivamb/exploratory-analysis-ga-customer-revenue
def getbin_hits(x):
    if x < 5:
        return "1-5"
    elif x < 10:
        return "5-10"
    elif x < 30:
        return "10-30"
    elif x < 50:
        return "30-50"
    elif x < 100:
        return "50-100"
    else:
        return "100+"

agg_dict = {}
for col in ["totals_bounces", "totals_hits", "totals_newVisits", "totals_pageviews", "totals_transactionRevenue"]:
    train[col] = train[col].astype('float')
    agg_dict[col] = "sum"
tmp = train.groupby("fullVisitorId").agg(agg_dict).reset_index()
tmp["total_hits_bin"] = tmp["totals_hits"].apply(getbin_hits)
tmp["totals_bounces_bin"] = tmp["totals_bounces"].apply(lambda x : str(x) if x <= 5 else "5+")
tmp["totals_pageviews_bin"] = tmp["totals_pageviews"].apply(lambda x : str(x) if x <= 50 else "50+")

t1 = tmp["total_hits_bin"].value_counts()
t2 = tmp["totals_bounces_bin"].value_counts()
t3 = tmp["totals_newVisits"].value_counts()
t4 = tmp["totals_pageviews_bin"].value_counts()

fig = tools.make_subplots(rows=2, cols=2, subplot_titles=["Total Hits per User", "Total Bounces per User", 
                                                         "Total NewVisits per User", "Total PageViews per User"], print_grid=False)

tr1 = go.Bar(x = t1.index[:20], y = t1.values[:20])
tr2 = go.Bar(x = t2.index[:20], y = t2.values[:20])
tr3 = go.Bar(x = t3.index[:20], y = t3.values[:20])
tr4 = go.Bar(x = t4.index, y = t4.values)

fig.append_trace(tr1, 1, 1)
fig.append_trace(tr2, 1, 2)
fig.append_trace(tr3, 2, 1)
fig.append_trace(tr4, 2, 2)

fig['layout'].update(height=700, showlegend=False)
iplot(fig)