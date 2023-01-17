%matplotlib inline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


import matplotlib.pyplot as plt
from plotnine import *
#import os
#print(os.listdir("../input"))
df = pd.read_csv('../input/ks-projects-201801.csv', index_col=0)
df.head()
plt.subplots(figsize=(19,5))
sns.countplot(df['main_category'])
sns.despine(bottom=True, left=True)
proportion_currency = df.groupby(['country'])['country'].count() / df['name'].count()
proportion_currency.sort_values(ascending=True).plot(kind='bar', title='% of campaigns by country', figsize=(10,5))
sns.jointplot(x='pledged', y='backers', data=df[(df['backers'] < 2500) & (df['backers'] > 500) & (df['pledged'] < 100000)  & (df['pledged'] > 10000)].sample(1000))
plt.subplots(figsize=(10,5))
plt.title('Campaign goals')
sns.kdeplot(df[(df['goal'] < 50000)]['goal'])
df.head()
plt.subplots(figsize=(20,5))
sns.violinplot(
    x='country',
    y='backers',
    data=df[(df['backers'] < 50000)].sample(50000)
)
#group by launch year
arr = [0,5, 10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000]
bins = pd.cut(df['backers'], arr )
success_bins = pd.cut(df[(df['state'] == 'successful')]['backers'], arr)
ax = (df[(df['state'] == 'successful')].groupby(success_bins)['backers'].agg(['count']) / df.groupby(bins)['backers'].agg(['count'])).plot(kind='barh', title='% of successful campaigns for the given backers', figsize=(10,10))
sns.despine(bottom=True, left=True)
# Credit for this snippet to Kromel, cheers : https://www.kaggle.com/kromel/kickstarter-successful-vs-failed

#group by launch year
main_colors = dict({'failed': 'rgb(200,50,50)', 'successful': 'rgb(50,200,50)'})

data = []
annotations = []
df['launched'] = pd.to_datetime(df['launched'])

dfSF = df[(df['state'] == 'successful') | (df['state'] == 'failed')]
rate_success_year = (dfSF[dfSF['state'] == 'successful'].groupby(dfSF.launched.dt.year)['launched'].count() / dfSF.groupby(dfSF.launched.dt.year)['launched'].count()) * 100
rate_failed_year = (dfSF[dfSF['state'] == 'failed'].groupby(dfSF.launched.dt.year)['launched'].count() / dfSF.groupby(dfSF.launched.dt.year)['launched'].count()) * 100


bar_success = go.Bar(
        x=rate_success_year.index,
        y=rate_success_year,
        name='successful',
        marker=dict(
            color=main_colors['successful'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

bar_failed = go.Bar(
        x=rate_failed_year.index,
        y=rate_failed_year,
        name='failed',
        marker=dict(
            color=main_colors['failed'],
            line=dict(
                color='rgb(100,100,100)',
                width=1,
            )
        ),
    )

data = [bar_success, bar_failed]
layout = go.Layout(
    barmode='stack',
    title='% of successful and failed projects by year',
    autosize=False,
    width=800,
    height=400,
    annotations=annotations
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='main_year')
df1 = df[(df['launched'].dt.year > 1970)]
df1_group_year = df1.groupby(df.launched.dt.year).mean()
df1_group_year['year'] = df1_group_year.index
(ggplot(df1_group_year)
     + aes(x='year', y='goal')
     + geom_col(fill='limegreen') 
     + ggtitle('Mean goal for campaigns each year')
     + xlab("Mean goal") + ylab("Year") + theme(figure_size=(5,5)))
df.head()
g = sns.FacetGrid(df[(df['goal'] < 50000)],col_wrap=6, col="main_category")
g.map(sns.kdeplot, "goal")
df_trimmed = df[(df['backers'] < 1000) & (df['goal'] < 50000) & (df['goal'] > 1000)]
df_trimmed_sf = df_trimmed[(df_trimmed['state'] == 'successful') | (df_trimmed['state'] == 'failed')]
(ggplot(df_trimmed_sf.sample(30000))
    + aes('goal', 'backers', color='state')
    + geom_point()
    + stat_smooth(method="lm")
    + facet_wrap('~main_category')
    + theme(figure_size=(20, 10),
            strip_background=element_rect(fill='white'),
            axis_line_x=element_line(color='white'),
            axis_line_y=element_line(color='white'),
            legend_key=element_rect(fill='white', color='white'))
)