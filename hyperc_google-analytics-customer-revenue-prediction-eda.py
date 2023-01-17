import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import datetime as dt
from pandas.io.json import json_normalize



from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
""" Format tables """

def format_table(df):
    df_formatted = df
    
    # Format date
    df_formatted["date"] = df_formatted["date"]\
        .apply(lambda x: dt.datetime.strptime(str(x),"%Y%m%d"))

    for dict_col_name in ["device", "totals", "trafficSource", "geoNetwork"]:
        df_formatted[dict_col_name] = df_formatted[dict_col_name]\
            .apply(lambda x: x.replace('true', 'True').replace('false', 'False'))\
            .apply(lambda x: dict(eval(x)))
        
    # Explode json columns
    for col_name in ["browser", "operatingSystem", "isMobile", "deviceCategory"]:
        df_formatted[col_name] = df_formatted["device"]\
            .apply(lambda x: x.get(col_name))

    for col_name in ["visits", "hits", "pageviews", "bounces", "newVisits", "transactionRevenue"]:
        df_formatted[col_name] = df_formatted["totals"]\
            .apply(lambda x: float(x.get(col_name) if x.get(col_name) else 0))

    for col_name in ["campaign", "source", "medium", "keyword"]:
        df_formatted[col_name] = df_formatted["trafficSource"]\
            .apply(lambda x: x.get(col_name))

    for col_name in ["continent", "subContinent", "country", "region"]:
        df_formatted[col_name] = df_formatted["geoNetwork"]\
            .apply(lambda x: x.get(col_name))

    df_formatted = df_formatted.drop(columns=["device", "totals", "trafficSource", "geoNetwork"])    
    
    # Add has_bought column
    df_formatted["has_bought"]=(df_formatted["transactionRevenue"]>0).apply(int)
    
    return df_formatted

train = format_table(train)

# A simple look at the train set
train.head()
print("Training set contains %s lines" % train.shape[0])
print("Probability: {}%".format(round(100 * train[train['transactionRevenue'] > 0].shape[0] / train.shape[0], 2)))
mean = train[train['transactionRevenue'] > 0]\
          .agg({'transactionRevenue': 'mean'})['transactionRevenue']

print("Mean basket price if payment: %s" % round(mean, 2))

median = train[train['transactionRevenue'] > 0]\
          .agg({'transactionRevenue': 'median'})['transactionRevenue']

print("Mean basket price if payment: %s" % round(median, 2))
# Count nb_sales
train_count = train\
    .groupby("date", as_index=False)\
    .agg({'visitId' : ['count'], 'transactionRevenue' : ['sum']})\
    .rename(columns={'visitId': 'nb_visits', 'transactionRevenue': 'total_revenue'})

train_count.columns=train_count.columns.droplevel(1)

train_count['revenue_per_visit'] = train_count['total_revenue'] / train_count['nb_visits']

# Add moving average
window_size = 7 # Weekly seasonality
train_count["visits_moving_average"] = train_count["nb_visits"].rolling(window_size, center=True).sum() / window_size
train_count["revenue_moving_average"] = train_count["total_revenue"].rolling(window_size, center=True).sum() / window_size
train_count["ratio_moving_average"] = train_count["revenue_per_visit"].rolling(window_size, center=True).sum() / window_size
# Prepare layout for plotting
layout = go.Layout(
    legend = dict(orientation="h")
)
#Create traces
layout.update({"title": "Evolution of number of daily visits over time"})

trace_daily = go.Scatter(
    x = train_count.date,
    y = train_count.nb_visits,
    name = 'nb_visits'
)
trace_average = go.Scatter(
    x = train_count.date,
    y = train_count.visits_moving_average,
    name = 'Moving average'
)

data = [trace_daily, trace_average]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')
layout.update({"title": "Evolution of daily revenues over time"})

trace_transactions = go.Scatter(
    x = train_count.date,
    y = train_count.total_revenue,
    name = 'Revenue'
)

trace_average = go.Scatter(
    x = train_count.date,
    y = train_count.revenue_moving_average,
    name = 'Moving average'
)

data = [trace_transactions, trace_average]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')
layout.update({"title": "Evolution of daily SumRevenue/NbVisits over time"})

#Create traces
trace_daily = go.Scatter(
    x = train_count.date,
    y = train_count.revenue_per_visit,
    name = 'Revenue per visit'
)
trace_average = go.Scatter(
    x = train_count.date,
    y = train_count.ratio_moving_average,
    name = 'Moving average'
)

data = [trace_daily, trace_average]
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')
train[(train['transactionRevenue'] > 0) & (train['transactionRevenue'] < 1000000000)]['transactionRevenue'].iplot(
    kind='histogram',
    barmode='stack',
    bins=1000,
    histnorm='probability',
    filename='cufflinks/customized-histogram',
    title='transactionRevenue distribution')
def plot_metrics_on_type(train, t):
    """ Top Part """
    # Daily visits evolutions
    df = train\
        .groupby(["date", t], as_index=False)\
        .visitId.agg(['count']).reset_index()\
        .pivot(index="date", columns=t, values="count")
    
    figure = df.iplot(kind='area', fill=True, asFigure=True)
    figure['layout'].update({'height': 400, 'paper_bgcolor':'#ffffff', 'title': 'Evolution of visits over time per %s' % t})
    py.iplot(figure)
    
    """ Bottom part """
    # Compare revenues and probability to buy 
    df_revenue_per_type = train[train["transactionRevenue"] > 0]\
        .groupby(t)\
        .agg({'transactionRevenue': 'mean'})
    
    df = train\
        .groupby([t], as_index=False)\
        .agg({'has_bought': 'mean'}).reset_index()\
        .merge(df_revenue_per_type, on=t)
    
    trace1 = go.Bar(
        x=df[t],
        y=df.has_bought,
        name='Has_bought'
    )
    trace2 = go.Bar(
        x=df[t],
        y=df.transactionRevenue,
        xaxis='x2',
        yaxis='y2',
        name='Mean revenue when a payment is done'
    )
    
    # Global repartition of visits
    df2 = train\
        .groupby([t], as_index=False)\
        .visitId.agg(['count']).reset_index()\
    
    trace3 = go.Bar(
        x=df2[t],
        y=df2["count"],
        name='Number of visits',
        xaxis='x3',
        yaxis='y3',
    )
    
    data = [trace3, trace1, trace2]
    layout = go.Layout(
        title="%s metrics" % t,
        showlegend=False,
        barmode='group',
        height=400,
        xaxis=dict(
            title='Probability of payment by %s' % t,
            domain=[0, 0.28],
            titlefont=dict(
                size=9,
            )
        ),
        yaxis=dict(
            domain=[0, 1]
        ),
        xaxis2=dict(
            title='Mean revenue by %s' % t,
            domain=[0.38, 0.62],
            titlefont=dict(
                size=9,
            )
        ),
        yaxis2=dict(
            domain=[0, 1],
            anchor='x2'
        ),
        xaxis3=dict(
            title='Number of visits by %s' % t,
            domain=[0.72, 1],
            titlefont=dict(
                size=9,
            )
        ),
        yaxis3=dict(
            domain=[0, 1],
            anchor='x3'
        )
    )
    
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
plot_metrics_on_type(train, "channelGrouping")
plot_metrics_on_type(train, "deviceCategory")
plot_metrics_on_type(train, "operatingSystem")
plot_metrics_on_type(train, "continent")
#plot_metrics_on_type(train, "country") #filter biggest countries first
df_payment_prob = train\
    .groupby("newVisits")\
    .agg({"visitId": "count", "has_bought": "mean"})\
    .reset_index()

df_transaction_mean_when_payment = \
    train[train["transactionRevenue"] > 0]\
        .groupby("newVisits")\
        .agg({"transactionRevenue": "mean"})\
        .reset_index()

trace1 = go.Bar(
    x=df_payment_prob["newVisits"],
    y=df_payment_prob["has_bought"],
    name='Payment probability'
)

trace2 = go.Bar(
    x=df_transaction_mean_when_payment["newVisits"],
    y=df_transaction_mean_when_payment["transactionRevenue"],
    xaxis='x2',
    yaxis='y2',
    name='Mean revenue when payment'
)

trace3 = go.Bar(
    x=df_payment_prob["newVisits"],
    y=df_payment_prob["visitId"],
    xaxis='x3',
    yaxis='y3',
    name='Number of visits'
)


layout = go.Layout(
    title="Analysis of newVisits field",
    showlegend=False,
    barmode='group',
    height=400,
    xaxis=dict(
        title='Probability of payment',
        domain=[0, 0.28]
    ),
    yaxis=dict(
        domain=[0, 1]
    ),
    xaxis2=dict(
        title='Mean revenue when a payment is done',
        domain=[0.38, 0.62]
    ),
    yaxis2=dict(
        domain=[0, 1],
        anchor='x2'
    ),
    xaxis3=dict(
        title='Number of visits',
        domain=[0.72, 1]
    ),
    yaxis3=dict(
        domain=[0, 1],
        anchor='x3'
    )
)

data = [trace1, trace2, trace3]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
train[(train['transactionRevenue'] < 1000000000)]\
    .groupby("visitNumber")\
    .agg({"has_bought": "mean"})\
    .iplot(kind='bar', title="Probability of payment given visitNumber field")
train[(train['transactionRevenue'] > 0) & (train['transactionRevenue'] < 1000000000)]\
    .groupby("visitNumber")\
    .agg({"transactionRevenue": "mean"})\
    .iplot(kind='bar', title="Mean transaction Revenue for visits that led to a payment by visitNumber field")
train\
    .groupby("hits")\
    .agg({'has_bought':'mean'})\
    .iplot(kind="bar", title="Probability of payment given the number of hits")
train[train["transactionRevenue"] > 0]\
    .groupby("hits")\
    .agg({'transactionRevenue':'mean'})\
    .iplot(kind="bar", title="Mean transactionRevenue for visits that led to payments, given the number of hits")
train\
    .groupby("pageviews")\
    .agg({'has_bought':'mean'})\
    .iplot(kind="bar", title="Probability of payment given the number of pageviews")
train[train["transactionRevenue"] > 0]\
    .groupby("pageviews")\
    .agg({'transactionRevenue':'mean'})\
    .iplot(kind="bar", title="Mean transactionRevenue for visits that led to payments, given the number of pageviews")
data = train[['pageviews','hits']]
data.corr(method='pearson')
print( "The test set contains %s lines" % test.shape[0])
common_ids = set(test["fullVisitorId"]).intersection(train["fullVisitorId"])
print( "%s visitors IDs are present in bot train and test set. (out of %s)" % (len(common_ids), len(set(test["fullVisitorId"]))))
