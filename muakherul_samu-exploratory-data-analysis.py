# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import sys
import datetime as dt
#from bqplot import pyplot as plt
#from wordcloud import WordCloud as wc
#from textblob import TextBlob
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('seaborn')
import os

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)


#style.use('fivethirtyeight')
%matplotlib inline
df = pd.read_csv('../input/fncln.csv',  error_bad_lines=False)
df.head()
## Simple function to convert bengali to english numerial digit. 
## same thing can be achieve with map() function
def bntoen(digit):
    converted = ""
    ban = ('০','১','২','৩','৪','৫','৬','৭','৮','৯')
    eng = ('0','1','2','3','4','5','6','7','8','9')
    for char in str(digit):
        if char in ban:
            converted+=eng[ban.index(char)]
    return converted

dff = ['৩৪৭৮৫৯:০২৩৪৫৬']
bntoen(dff)
## apply function to views, like and comment column
df.views = df.views.apply(lambda x: bntoen(x))
df.like = df.like.apply(lambda x: bntoen(x))
df.comment = df.comment.apply(lambda x: bntoen(x))
df.head()
#rename date column to date_bn to use 'date' name in future
df.rename(columns={'date':'date_bn'}, inplace=True)
#split date_bn to four columns, named respectively..
df[['1','2','3','4']] = df.date_bn.str.split(' ', expand=True)
df.head()
## simple function to map 3 to new mm column
month = {'জানুয়ারি': '1','ফেব্রুয়ারি': '2','মার্চ': '3','এপ্রিল': '4','মে': '5','জুন': '6','জুলাই': '7','আগস্ট': '8','সেপ্টেম্বর': '9','অক্টোবর': '10', 'নভেম্বর': '11', 'ডিসেম্বর': '12'}
df['mm'] = df['3'].map(month)
## map 1 and 4 column to new dd and yyyy column for day and year value
df['dd'] = df['1'].apply(lambda x: bntoen(x))
df['yyyy'] = df['4'].apply(lambda x: bntoen(x))
## create new date column with mapped day, month and year value
df['date'] = df.dd.astype(str) + '-' + df.mm.astype(str) + '-' + df.yyyy.astype(str)
## Renaming time to time_bn for future use
df.rename(columns={'time':'time_bn'}, inplace=True)
## Split time_bn column
df[['frmt', 'tt']] = df.time_bn.str.split(' ', expand=True)
df[['hr', 'mn']] = df.tt.str.split(':', expand=True)
df.head()
## Map to english digit
df.hr = df.hr.apply(lambda x: bntoen(x))
df.mn = df.mn.apply(lambda x: bntoen(x))
## Map all values except 'রাত'
frmt_map = {'দুপুর': 'PM', 'বিকাল': 'PM', 'ভোর': 'AM', 'সকাল': 'AM', 'সন্ধ্যা': 'PM' }
df.frmt = df.frmt.map(frmt_map)
## New column for mapping valur 'রাত'
df['frmt_new'] = np.where(df.frmt.isnull(), np.where(df.hr == '12', 'AM', np.where(df.hr == '1', 'AM', np.where(df.hr == '2', 'AM', np.where(df.hr == '3', 'AM', 'PM')))), 'PM')
## Now fill nan values from frmt_new to frmt
df.frmt = df.frmt.fillna(df.frmt_new)
## Join day month and year to new column
df['time'] = df.hr.astype(str) + ':' + df.mn.astype(str) + ' ' + df.frmt.astype(str)
## date and time join 
df['datetime'] = df.date.astype(str) + ' ' + df.time.astype(str)
## date had some nan value, getting rid of them
df = df[df.datetime != '-nan- : PM']
## Pandas datetime formatting
df.datetime = pd.to_datetime(df.datetime, format='%d-%m-%Y %I:%M %p')
## Select relevant data columns for dataframe
df = df[['title', 'author', 'author_id', 'views', 'comment', 'like', 'datetime']]
df.head()
for i in range(len(df.loc[0,:])):
    print('NaN values in ' + str(df.columns[i]) + ' : ' + str(len(df[df[df.columns[i]].isnull() == True])))
## Change object type to integer
df.views = df.views.astype(int)
df.comment = df.comment.astype(int)
df.like = df.like.astype(int)
df.info()
print('Total number of post: ' + str(len(df.title)))
print('Toral number of bloggers: ' + str(len(df.author_id.unique())))
print('Total number of views: '+ str(sum(df.views)))
print('Total number of comments: ' + str(sum(df.comment)))
print('Timeline: ' + str((df.datetime.max()-df.datetime.min()).days) + ' days')
## Sort values according to datetime
df = df.sort_values('datetime')
df_plot = df.datetime.groupby(df.datetime.dt.year).agg(len)
ax = df_plot.plot(kind = 'barh',alpha=.88,fontsize='13',width=0.8, figsize = (10,8))

plt.ylabel('posts')

for i, v in enumerate(df.datetime.groupby(df.datetime.dt.year).agg(len)):
    ax.text(v-20, i -.15, str(v), fontweight='bold')

plt.title('Posts per year', fontsize='15')
plt.tight_layout()
data1 = df.reset_index().set_index('datetime')['views'].resample('M').count()

#fig, ax = plt.subplots()
data1.plot(figsize = (25,9), fontsize='18')
#plt.tight_layout()
plt.title('Post per Month', fontsize='24')
plt.tight_layout()
#df.reset_index().set_index('datetime')['views'].resample('D').count().plot(figsize = (40,16),color='coral', fontsize='28')
#plt.tight_layout()
#plt.title('Post per day', fontsize='36')
#df.pivot_table('views', index='datetime', aggfunc='count').plot(kind='bar')
#Prepare date 
x = df.groupby(df.datetime.dt.date)['views'].count()
type(x)
y = x.to_frame('views').reset_index()
#data = [go.Scatter(x=y.datetime, y=y.views)]
#iplot(data)
trace = go.Scatter(x=y.datetime,
                   y=y.views)

data = [trace]
layout = dict(
    title='Number of Post per Day',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig)
x = df.groupby(df.datetime.dt.date)['views'].sum()
type(x)
y = x.to_frame('views').reset_index()


trace = go.Scatter(x=y.datetime,
                   y=y.views)

data = [trace]
layout = dict(
    title='Number of Post per Day',
    xaxis=dict(zeroline=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig)
df.sort_values('views', ascending=False).head(20)
df.sort_values('comment', ascending=False).head(20)

x = df.groupby('author')['views'].sum().sort_values(ascending=False).head(30)
#type(x)
y = x.to_frame('views').reset_index()
data = [go.Bar(x=y.author, y=y.views)]
iplot(data)
x = df.groupby('author')['views'].count().sort_values(ascending=False).head(30)
y = x.to_frame('views').reset_index()
data = [go.Bar(x=y.author, y=y.views)]
iplot(data)
