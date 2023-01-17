import os
import numpy as np
import pandas as pd
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

path = "../input/python-quick-start-clean-and-pickle-data"
source_cols = ['trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType',
               'trafficSource_adwordsClickInfo.gclId', 'trafficSource_adwordsClickInfo.isVideoAd', 
               'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot', 
               'trafficSource_campaign', 'trafficSource_isTrueDirect',
               'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_referralPath', 'trafficSource_source']
train = pd.read_pickle(os.path.join(path, 'train_clean.pkl'))
test = pd.read_pickle(os.path.join(path, 'test_clean.pkl'))
train[source_cols].head()    # Head of traffic source columns
def plotbar(df, col, title, top=None):
    frame = pd.DataFrame()
    frame['totals_transactionRevenue'] = df['totals_transactionRevenue'].copy()
    frame[col] = df[col].fillna('missing')
    # Percentage of revenue
    tmp_rev = frame.groupby(col)['totals_transactionRevenue'].sum().to_frame().reset_index()
    tmp_rev = tmp_rev.sort_values('totals_transactionRevenue', ascending=False)
    tmp_rev = tmp_rev.rename({'totals_transactionRevenue': 'Revenue percentage'},axis=1)
    tmp_rev['Revenue percentage'] = 100*tmp_rev['Revenue percentage']/df['totals_transactionRevenue'].sum()
    # Percentage of visits
    tmp = frame[col].value_counts().to_frame().reset_index()
    tmp.sort_values(col, ascending=False)
    tmp = tmp.rename({'index': col, col: 'Percentage of Visits'},axis=1)
    tmp['Percentage of Visits'] = 100*tmp['Percentage of Visits']/len(df)
    tmp = pd.merge(tmp, tmp_rev, on=col, how='left')
    if top:
        tmp = tmp.head(top)
    # Barplot
    trace1 = go.Bar(x=tmp[col], y=tmp['Percentage of Visits'],
                    name='Visits', marker=dict(color='rgb(55, 83, 109)'))
    trace2 = go.Bar(x=tmp[col], y=tmp['Revenue percentage'],
                    name='Revenue', marker=dict(color='rgb(26, 118, 255)'))

    layout = go.Layout(
        barmode='group',
        title=title,
    )
    
    layout = go.Layout(
        title=title,
        xaxis=dict(tickfont=dict(size=14, color='rgb(107, 107, 107)')),
        yaxis=dict(
            title='Percentage',
            titlefont=dict(size=16, color='rgb(107, 107, 107)'),
            tickfont=dict(size=14, color='rgb(107, 107, 107)')
        ),
        legend=dict(x=0.95, y=1.0, bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)
    
plotbar(train, 'trafficSource_medium', 'Train set - visits and revenue by medium')
plotbar(train[train['trafficSource_medium'] == 'organic'], 'trafficSource_source', 'Visits and revenue by source for Organic search')
def print_value_counts(category):
    cols = ['trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType',
           'trafficSource_adwordsClickInfo.gclId', 'trafficSource_adwordsClickInfo.isVideoAd',
           'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot',
           'trafficSource_referralPath', 'trafficSource_campaign']
    for c in cols:
        nunique = train[train['trafficSource_medium'] == category][c].nunique()
        if nunique < 5:
            print(train[train['trafficSource_medium'] == category][c].value_counts(dropna=False))
        else:
            print(train[train['trafficSource_medium'] == category][c].describe())
            
print_value_counts('organic')
plotbar(train[train['trafficSource_medium'] == 'referral'], 'trafficSource_source', 'Visits and revenue by source for Referral', top=8)
print_value_counts('referral')
plotbar(train[train['trafficSource_medium'] == 'referral'], 'channelGrouping', 'Visits and revenue by channel for REFERRAL MEDIUM')
plotbar(train[train['trafficSource_medium'] == 'cpc'], 'trafficSource_source', 'Visits and revenue by source for CPC', top=None)
print_value_counts('cpc')
plotbar(train[train['trafficSource_medium'] == 'cpc'], 'trafficSource_campaign', 'Visits and revenue by campaign for CPC')
plotbar(train[train['trafficSource_medium'] == 'cpm'], 'trafficSource_source', 'Visits and revenue by source for CPM')
print_value_counts('cpm')
#print(train[train['trafficSource_medium'] == 'affiliate']['trafficSource_campaign'].value_counts())
print_value_counts('affiliate')
print_value_counts('(none)')