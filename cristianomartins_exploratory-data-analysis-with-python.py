import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tools
import matplotlib.pyplot as plt
import colorlover as cl
from IPython.display import HTML
import calendar
import datetime
from datetime import datetime, timedelta
init_notebook_mode(connected=True)

playstore_df = pd.read_csv('../input/googleplaystore.csv')
playstore_reviews_df = pd.read_csv('../input/googleplaystore_user_reviews.csv')
playstore_df.head()
playstore_reviews_df.head()
playstore_df.dropna(inplace=True)
playstore_df = playstore_df[playstore_df.Installs != 'Free']
category_install_df = pd.DataFrame(playstore_df, columns=['Category','Installs'])
category_install_df['Installs'] = category_install_df.Installs.apply(lambda x: x.replace('+','')).apply( lambda x: float(x.replace(',','')))
category_install_df = category_install_df.groupby('Category').sum().sort_values(by='Installs', ascending=False)
category_install_df.head()
colors_scale = cl.scales['10']['div']['RdYlGn']
colors = cl.interp( colors_scale, 40 ) 

trace = go.Bar(
    x = category_install_df.index,
    y = category_install_df['Installs'],
    name = 'Installs by Categories',
    marker={'color': cl.to_rgb( colors )}
   
)

data = [trace]
layout = go.Layout(
    title='<b>Installs by Categories<b>',
     margin=go.layout.Margin(
        l=173,
        r=80,
        t=100,
        b=200,
        pad=4
    ),
    xaxis=dict(
        title='<b>Category<b>',
        tickangle=-45,
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        ),
    ),
    yaxis=dict(
        title='<b>Installs<b>',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        ),
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
totalapp_free_df = (playstore_df[playstore_df.Type == 'Free'].count().iloc[0] / playstore_df['App'].count() * 100).round()
totalapp_paid_df = (playstore_df[playstore_df.Type == 'Paid'].count().iloc[0] / playstore_df['App'].count() * 100).round()
labels = ['Free','Paid']
values = [totalapp_free_df, totalapp_paid_df]

trace = go.Pie(
    labels=labels, 
    values=values,
    hoverinfo='label+percent', textinfo='percent', 
    textfont=dict(size=20, color='white')
)

data = [trace]

layout = go.Layout(
    title='<b>Distribution by App Type <b>',
     margin=go.layout.Margin(
        l=173,
        r=80,
        t=100,
        b=80,
        pad=4
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
android_versions = playstore_df.groupby(by='Android Ver').size().reset_index(name='total').sort_values(by='total', ascending=True)
colors_scale = cl.scales['7']['seq']['YlGnBu']
colors = cl.interp( colors_scale, 40 ) 

trace = go.Bar(
    x = android_versions.total,
    y = android_versions['Android Ver'],
    orientation = 'h',
    marker={'color': cl.to_rgb( colors )}
)

data = [trace]
layout = go.Layout(
    title='<b>App by Android Version<b>',
    xaxis=dict(
        title='<b>Total<b>',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='<b>Versions<b>',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    margin=dict(
        l=173,
        r=80,
        t=100,
        b=80,
        pad=4

    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
def getTotalDays(date):
    datetime_obj = datetime.strptime(date, '%B %d, %Y')
    days = datetime_obj - datetime.now() 
    return abs(days.days)

app_lastudpate_df = playstore_df.loc[:,['App','Type','Last Updated']]
app_lastudpate_df['day'] = app_lastudpate_df['Last Updated'].apply(lambda x: datetime.strptime(x, '%B %d, %Y').day)
app_lastudpate_df['month'] = app_lastudpate_df['Last Updated'].apply(lambda x: datetime.strptime(x, '%B %d, %Y').month)
app_lastudpate_df['year'] = app_lastudpate_df['Last Updated'].apply(lambda x: datetime.strptime(x, '%B %d, %Y').year)
app_lastudpate_df['total_days'] = app_lastudpate_df['Last Updated'].apply(lambda x: getTotalDays(x))
app_lastudpate_df.sort_values(by='total_days', ascending=False)
no_updates_df = app_lastudpate_df.loc[:,['App','Type','Last Updated','year','total_days']]
no_updates_df = no_updates_df.sort_values(by='total_days', ascending=False)
apps_updated_by_year_df = no_updates_df.groupby(by=['Type','year']).size().reset_index(name='total').sort_values(by='year', ascending=True)
trace_free = go.Bar(
    x = apps_updated_by_year_df[apps_updated_by_year_df.Type == 'Free'].year,
    y = apps_updated_by_year_df[apps_updated_by_year_df.Type == 'Free'].total,
    name = 'Free'
)

trace_paid = go.Bar(
    x = apps_updated_by_year_df[apps_updated_by_year_df.Type == 'Paid'].year,
    y = apps_updated_by_year_df[apps_updated_by_year_df.Type == 'Paid'].total,
    name = 'Paid'
)

layout = go.Layout(
    title='Total Apps Last update by Year',
    barmode='group',
    xaxis=dict(
        title='Years',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='Total Apps Last Updated',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    )
)

data = [trace_free, trace_paid]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
category_rating_df = playstore_df.groupby(by=['Category','Rating']).size().reset_index(name='total').sort_values(by='Rating', ascending=False)
category_rating_5_stars = category_rating_df[category_rating_df.Rating == 5]
category_rating_4_stars = category_rating_df[category_rating_df.Rating == 4]
category_rating_3_stars = category_rating_df[category_rating_df.Rating == 3]
category_rating_2_stars = category_rating_df[category_rating_df.Rating == 2]
category_rating_1_stars = category_rating_df[category_rating_df.Rating == 1]
trace_5_stars = go.Bar(
    x = category_rating_5_stars.Category,
    y = category_rating_5_stars.total,
    name = 'Rating 5 Stars',
    marker=dict(
        color='rgba(102.0, 194.0, 165.0, 0.7)',
        line=dict(
            color='rgba(102.0, 194.0, 165.0, 1.0)',
        )
    )
)

trace_4_stars = go.Bar(
    x = category_rating_4_stars.Category,
    y = category_rating_4_stars.total,
    name = 'Rating 4 Stars',
    marker=dict(
       color='rgba(252.0, 141.0, 98.0, 0.7)',
       line=dict(
           color='rgba(252.0, 141.0, 98.0, 1.0)',
       )
    )
)

trace_3_stars = go.Bar(
    x = category_rating_3_stars.Category,
    y = category_rating_3_stars.total,
    name = 'Rating 3 Stars',
    marker=dict(
       color='rgba(141.0, 160.0, 203.0)',
       line=dict(
           color='rgba(141.0, 160.0, 203.0)',
       )
    )
)

trace_2_stars = go.Bar(
    x = category_rating_2_stars.Category,
    y = category_rating_2_stars.total,
    name = 'Rating 2 Stars',
    marker=dict(
       color='rgba(231.0, 138.0, 195.0)',
       line=dict(
           color='rgba(231.0, 138.0, 195.0)',
       )
    )
)

trace_1_stars = go.Bar(
    x = category_rating_1_stars.Category,
    y = category_rating_1_stars.total,
    name = 'Rating 1 Stars',
    marker=dict(
        color='rgba(166.0, 216.0, 84.0, 0.7)',
        line=dict(
            color='rgba(166.0, 216.0, 84.0, 1.0)',
        )
    )
)


layout = go.Layout(
    title='<b>Rating by Categories<b>',
    barmode='stack',
    xaxis=dict(
        title='<b>Category<b>',
        tickangle=-45,
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='<b>Total<b>',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    margin=dict(
        l=173,
        r=80,
        t=100,
        b=200,
        pad=4

    )
)

data = [trace_5_stars, trace_4_stars, trace_3_stars, trace_2_stars, trace_1_stars]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)

positive_review_df = (playstore_reviews_df[playstore_reviews_df.Sentiment == 'Positive'].count().iloc[0] / playstore_reviews_df['Sentiment'].count() * 100).round()
negative_review_df = (playstore_reviews_df[playstore_reviews_df.Sentiment == 'Negative'].count().iloc[0] / playstore_reviews_df['Sentiment'].count() * 100).round()
neutral_review_df = (playstore_reviews_df[playstore_reviews_df.Sentiment == 'Neutral'].count().iloc[0] / playstore_reviews_df['Sentiment'].count() * 100).round()
colors = cl.scales['3']['qual']['Paired']

labels = ['Positive','Negative','Neutral']
values = [positive_review_df, negative_review_df, neutral_review_df]

trace = go.Pie(
    labels=labels, 
    values=values,
    hoverinfo='label+percent', textinfo='percent', 
    textfont=dict(size=20, color='white'),
    marker=dict(colors=cl.to_rgb( colors ))
    
)

data = [trace]

layout = go.Layout(
    title='<b>Distribution by Sentiment <b>',
     margin=go.layout.Margin(
        l=173,
        r=80,
        t=100,
        b=80,
        pad=4
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)