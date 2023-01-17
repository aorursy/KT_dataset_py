# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
%matplotlib inline



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_oakland_raw = pd.read_csv('../input/service-requests-received-by-the-oakland-call-center.csv')
df_oakland = df_oakland_raw.copy()
df_oakland.head(10)
df_oakland.dtypes
# Parse dates
df_oakland['DATETIMEINIT'] = pd.to_datetime(df_oakland_raw['DATETIMEINIT'])
df_oakland['DATETIMECLOSED'] = pd.to_datetime(df_oakland_raw['DATETIMECLOSED'])
# Replace instances of  STATUS  "Cancel" with CANCEL
df_oakland['STATUS'].replace('Cancel', 'CANCEL', inplace=True)
# Clean up district: replace lower case with upper, and a 4 with CCD4
df_oakland['COUNCILDISTRICT'].replace('4', 'CCD4', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd1', 'CCD1', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd2', 'CCD2', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd3', 'CCD3', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd4', 'CCD4', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd5', 'CCD5', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd6', 'CCD6', inplace=True)
df_oakland['COUNCILDISTRICT'].replace('ccd7', 'CCD7', inplace=True)

def get_coordinate(row, col_name):
    """ extracts given col_name from dictionary in a try, except block (some rows might not have lat/long)"""
    try:
        coordinate = row[col_name]
    except KeyError:
        coordinate = np.nan
    return coordinate
# Create some columns

# Latitude and longitude
df_oakland['address'] = df_oakland_raw['REQADDRESS'].map(ast.literal_eval)
df_oakland['LATITUDE'] = df_oakland['address'].apply(get_coordinate, args=('latitude', ))
df_oakland['LONGITUDE'] = df_oakland['address'].apply(get_coordinate, args=('longitude', ))
df_oakland.drop('address', axis=1, inplace=True)

# time open (duration, in days)
df_oakland['DURATION'] = df_oakland['DATETIMECLOSED'] - df_oakland['DATETIMEINIT']
df_oakland['DURATION'] = df_oakland['DURATION'].dt.days

# If still open, calculate DURATION up to current date 
today = pd.Timestamp.today()
df_oakland.loc[df_oakland['STATUS'] == 'OPEN', 'DURATION'] = (today - df_oakland['DATETIMEINIT']).dt.days

# Date column, no time
df_oakland['DATEINIT'] = df_oakland['DATETIMEINIT'].dt.date
# plot number of issues open per day
def plot_bar(df, days=None, title='', yaxis='', xaxis=''):
    """
    Plots columns as stacked bar chart, assumes dates in index
    """
    
    # filter df to number of given days
    df = df.sort_index()
    if days:
        cutoff = df.index.max()-datetime.timedelta(days=days)
        df = df.loc[df.index > cutoff]
    data = []
    for col in df.columns:
        trace = go.Bar(
                    x = df.index,
                    y = df[col],
                    name=col
                    )
        data.append(trace)

    layout = go.Layout(
                barmode = 'stack',
                title=title,
                yaxis=dict(
                title=yaxis,
                )
                )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    

def plot_issues(df, col, days=None, title='', yaxis='', xaxis=''):
    """group df by a column and plot stacked bar chart"""
    df_group = df.groupby(['DATEINIT', col])[col].count().unstack(col)
    plot_bar(df_group, days=days, title=title, yaxis=yaxis, xaxis='')
# By status
#sns.countplot(data=df_oakland, y='STATUS').set_title('Number of issues by status')
# Plot issues opened in last month by category
plot_issues(df_oakland, 'REQCATEGORY', days=30, title='Daily issues opened over the last 30 days, by category')
# What happened on feb 26? mass dump of tickets?
# Plot issues by district
plot_issues(df_oakland, 'COUNCILDISTRICT', days=30, title='Daily issues opened over the last 30 days, by district')
# Open issues by district
sns.countplot(y=df_oakland.loc[df_oakland['STATUS'] == 'OPEN','COUNCILDISTRICT']).set_title('Currently open issues by district')
def plot_duration(df, category, months=6, title='', h=True):
    today = pd.Timestamp.today()
    date_cutoff = today - pd.DateOffset(months=months)
    duration_mask = df.loc[df['DATETIMEINIT'] >= date_cutoff]
    
    if h:
        chart = duration_mask.groupby(category)['DURATION'].mean().sort_values().plot.barh(color='#112F41')
        chart.set_xlabel('days')
    else:
        chart = duration_mask.groupby(category)['DURATION'].mean().sort_values(ascending=False).plot.bar(color='#112F41')
        chart.set_ylabel('days')
    chart.set_title(title)
    return chart 
category = 'COUNCILDISTRICT'
months = 6
plot_duration(df_oakland, category, months=months, title=f'Average duration of issues opened over the last {months} months, by {category}', h=True)
category = 'REQCATEGORY'
months = 6
plot_duration(df_oakland, category, months=months, title=f'Average duration of issues opened over the last {months} months, by {category}', h=False)
# only use last 6 months
today = pd.Timestamp.today()
months = 3
date_cutoff = today - pd.DateOffset(months=months)
duration_mask = df_oakland.loc[df_oakland['DATETIMEINIT'] >= date_cutoff]
df = duration_mask.sort_values('COUNCILDISTRICT')


data = []
for district in df['COUNCILDISTRICT'].unique():
    trace = go.Box(
        y=df.loc[df['COUNCILDISTRICT'] == district, 'DURATION'],
        boxpoints='suspectedoutliers',
        name=district,
    )
    data.append(trace)
    
layout = go.Layout(
    title='Distribution of the duration by district, for issues open in the last 3 months',
    yaxis = dict(
        title='days'
        )
    )

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig)
#df.loc[df['COUNCILDISTRICT'] == 'CCD4'].shape