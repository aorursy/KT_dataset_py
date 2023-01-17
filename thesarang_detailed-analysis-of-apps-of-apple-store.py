import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from plotly import tools
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()

import warnings
warnings.filterwarnings('ignore') 

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8') #

%matplotlib inline
df = pd.read_csv('../input/AppleStore.csv', encoding='utf8')
df.drop(axis=1, labels=['Unnamed: 0'], inplace=True) #droping unwanted 'Unnamed: 0' column.
df.head() 
df.info() #Information about columns
df.shape #data has 7197 rows with 16 features
df.isnull().any()
def bytes_convertor(B):
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    if B < KB:
        return '{0:.2f}Bytes'.format(B)
    elif KB <= B < MB:
        return '{0:.2f}KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f}MB'.format(B/MB)
    elif GB <= B:
        return '{0:.2f}GB'.format(B/GB)

df['app_size'] = df['size_bytes'].apply(bytes_convertor)
df = df[['id', 'track_name', 'size_bytes', 'app_size','currency', 'price',
       'rating_count_tot', 'rating_count_ver', 'user_rating',
       'user_rating_ver', 'ver', 'cont_rating', 'prime_genre',
       'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic']]
large_size = df[['track_name','size_bytes', 'app_size']].sort_values(by='size_bytes', ascending=False)[0:15]
large_size['app_size'] = large_size['app_size'].str.replace('GB','')
large_size['app_size'] = large_size['app_size'].apply(pd.to_numeric)
trace = go.Bar(
    x = large_size.track_name,
    y = large_size.app_size,
    marker = dict(
        color = 'rgb(242, 215, 213)',
        line = dict(
            color = 'rgb(100, 30, 22)',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    title = 'Top 15 Biggest games',
    xaxis = dict(
        title = 'Games'
    ),
    yaxis = dict(
        title = 'Size of the games in GBs'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
small_size = df[['track_name','size_bytes', 'app_size']].sort_values(by='size_bytes', ascending=True)[:15]
small_size['app_size'] = small_size['app_size'].str.replace('KB','')
small_size['app_size'] = small_size['app_size'].apply(pd.to_numeric)
trace = go.Bar(
    y = small_size.app_size,
    x = small_size.track_name,
    marker = dict(
        color = 'rgb(215, 189, 226)',
        line = dict(
            color = 'rgb(108, 52, 131)',
            width = 1
        )
    )
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Games'
    ),
    yaxis = dict(
        title='Size of games in KBs'
    ),
    title = 'Top 15 Smallest games',
    margin = dict(
        l = 100,
        t = 100
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
free_or_paid = df['price'].apply(lambda x:1 if x > 0 else 0)
free_or_paid = free_or_paid.value_counts().reset_index()
free_or_paid.rename(columns={'index':'free or paid', 'price':'count'}, inplace=True)
free_or_paid.loc[free_or_paid['free or paid'].values == 0, 'free or paid'] = 'free' #THERE ARE 4056 GAMES WHICH ARE FREE
free_or_paid.loc[free_or_paid['free or paid'].values == 1, 'free or paid'] = 'paid' #THERE ARE 3141 GAMES WHICH ARE PAID
trace = go.Bar(
    x = free_or_paid['free or paid'],
    y = free_or_paid['count'],
    marker = dict(
        color = 'pink',
        line = dict(
            color = 'red',
            width = 1
        )
    )
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Free or paid'
    ),
    yaxis = dict(
        title = 'Number of apps'
    ),
    title = 'Number of free vs paid apps',
    margin = dict(
        l = 100,
        t = 100
    )
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
app_genre = df.prime_genre.value_counts().reset_index()[0:25]
app_genre.rename(columns={'index':'genre', 'prime_genre':'count'}, inplace=True)
trace = go.Bar(
    x = app_genre['genre'],
    y = app_genre['count'],
    marker = dict(
        color = 'lightgreen',
        line=dict(
            color='green',
            width=1.5),
    ),
    
)

data = [trace]

layout = go.Layout(
    margin = dict(
        l = 100,
        t = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Number of apps'
    ),
    title = 'Total number of apps with genres',
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
top_rating_apps = df[['track_name','app_size', 'rating_count_tot']].sort_values(by='rating_count_tot', ascending=False)[0:20]

trace = go.Bar(
    x = top_rating_apps['track_name'],
    y = top_rating_apps['rating_count_tot'],
    name = 'Highest rating apps',
    marker = dict(
        color = 'rgb(169, 204, 227)',
        line=dict(
            color='rgb(21, 67, 96)',
            width=1.5),
    ),
)

data = [trace]

layout = go.Layout(
    margin = dict(
        l = 100,
        t = 100
    ),
    xaxis = dict(
        title = 'Apps name'
    ),
    yaxis = dict(
        title = 'Total user rating'
    ),
    title = 'Top user rated apps'
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
user_rating_ver = df.user_rating.value_counts().reset_index()
user_rating_ver.rename(columns={'index':'ratings', 'user_rating':'count'}, inplace=True)

trace = go.Bar(
    x = user_rating_ver['ratings'],
    y = user_rating_ver['count'],
    name = 'user ratings',
    marker = dict(
        color = 'rgb(163, 228, 215)',
        line=dict(
            color='rgb(14, 98, 81)',
            width=1),
    ),
)

data = [trace]

layout = go.Layout(
    margin = dict(
        l = 100,
        t = 100
    ),
    xaxis = dict(
        title = 'Average user rating' 
    ),
    
    yaxis = dict(
        title = 'Number of apps'
    ),
    title = 'Top average user rated apps',
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
cont_rating = df['cont_rating'].value_counts().reset_index()
cont_rating.rename(columns={'index':'age', 'cont_rating':'count'}, inplace=True)
cont_rating


trace = go.Bar(
    x = cont_rating['age'],
    y = cont_rating['count'],
    marker = dict(
        color = 'lightgreen',
        line = dict(
            color = 'green',
            width = 1
        )
    )
)

data = [trace]

layout = go.Layout(
    margin = dict(
        l = 100,
        t = 100
    ),
    xaxis = dict(
        title = 'Content ratings'
    ),
    yaxis = dict(
        title = 'Number of apps'
    ),
#     plot_bgcolor = 'black',
#     paper_bgcolor = 'black',
    title = 'Total apps with contant ratings'
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
supporting_devices = df['sup_devices.num'].value_counts().reset_index()
supporting_devices.rename(columns={'index':'supporting_devices', 'sup_devices.num':'count'}, inplace= True)
supporting_devices.sort_values(by='supporting_devices', ascending = False)

trace = go.Bar(
    x = supporting_devices['supporting_devices'],
    y = supporting_devices['count'],
    marker = dict(
        color = 'pink',
        line = dict(
            color = 'red',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Number of supporting devices'
    ),
    yaxis = dict(
        title = 'Number of apps'
    ),
    title = 'Number of devices supported by apps',
    margin = dict(
        l = 100,
        t = 100
    )
    
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
supporting_languages = df[['track_name','lang.num']].sort_values(by='lang.num', ascending=False)[0:20]

trace = go.Bar(
    x = supporting_languages['track_name'],
    y = supporting_languages['lang.num'],
    marker = dict(
        color = 'grey',
        line = dict(
            color = 'black',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'App Name'
    ),
    yaxis = dict(
        title = 'Number of languages'
    ),
    title = 'Total number of Languages supported by apps',
    margin = dict(
        l = 100,
        t = 100
    )
    
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
genrewise_p = df[['track_name','price','rating_count_tot', 'prime_genre']]
genrewise_p = genrewise_p[genrewise_p['price'] != 0].sort_values(by='rating_count_tot', ascending=False)
genrewise_p = genrewise_p[['prime_genre','track_name']].groupby(['prime_genre']).count().sort_values(by='track_name', ascending=False).reset_index()
genrewise_p.rename(columns={'track_name':'count'}, inplace=True)
genrewise_f = df[['track_name','price','rating_count_tot', 'prime_genre']]
genrewise_f = genrewise_f[genrewise_f['price'] == 0].sort_values(by='rating_count_tot', ascending=False)
genrewise_f = genrewise_f[['prime_genre','track_name']].groupby(['prime_genre']).count().sort_values(by='track_name', ascending=False).reset_index()
genrewise_f.rename(columns={'track_name':'count'}, inplace=True)
trace1 = go.Bar(
    x = genrewise_p['prime_genre'],
    y = genrewise_p['count'],
    marker = dict(
        color = 'lightgreen',
        line = dict(
            color = 'green',
            width = 1
        )
    )
)

trace2 = go.Bar(
    x = genrewise_f['prime_genre'],
    y = genrewise_f['count'],
    marker = dict(
        color = 'lightblue',
        line = dict(
            color = 'blue',
            width = 1
        )
    )
)

trace3 = go.Bar(
    x = genrewise_p['prime_genre'],
    y = genrewise_p['count'],
    marker = dict(
        color = 'lightgreen',
        line = dict(
            color = 'green',
            width = 1
        )
    )
)

trace4 = go.Bar(
    x = genrewise_f['prime_genre'],
    y = genrewise_f['count'],
    marker = dict(
        color = 'lightblue',
        line = dict(
            color = 'blue',
            width = 1
        )
    )
)

fig = tools.make_subplots(rows=2, cols=2, specs = [[{}, {}], [{'colspan':2}, None]], 
                          subplot_titles=('(1) Each genre count in paid apps', '(2) Each genre count in free apps', 
                                          '(3) Grouped barplot containing count of Free(Blue) and Paid(Green) apps'), print_grid=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 1)

fig['layout']['yaxis1'].update(title='Number of apps')
fig['layout']['yaxis2'].update(title='Number of apps')
fig['layout']['yaxis3'].update(title='Number of apps')
fig['layout'].update(showlegend=False, height=800)

py.iplot(fig)
games = df[df['prime_genre'] == 'Games'].sort_values(by='rating_count_tot', ascending=False)[0:15]
entertainment = df[df['prime_genre'] == 'Entertainment'].sort_values(by='rating_count_tot', ascending=False)[0:15]
photo_video = df[df['prime_genre'] == 'Photo & Video'].sort_values(by='rating_count_tot', ascending=False)[0:15]
education = df[df['prime_genre'] == 'Education'].sort_values(by='rating_count_tot', ascending=False)[0:15]
trace1 = go.Scatter(
    x = games['track_name'],
    y = games['rating_count_tot'],
    marker = dict(
        color = 'lightblue',
        line = dict(
            color = 'blue',
            width = 1
        )
    )
)

trace2 = go.Scatter(
    x = entertainment['track_name'],
    y = entertainment['rating_count_tot'],
    marker = dict(
        color = 'pink',
        line = dict(
            color = 'red',
            width = 1
        )
    ),
)

trace3 = go.Scatter(
    x = photo_video['track_name'],
    y = photo_video['rating_count_tot'],
    marker = dict(
        color = 'lightgreen',
        line = dict(
            color = 'green',
            width = 1
        )
    )
)

trace4 = go.Scatter(
    x = education['track_name'],
    y = education['rating_count_tot'],
    marker = dict(
        color = 'grey',
        line = dict(
            color = 'black',
            width = 1
        )
    )
)

fig = tools.make_subplots(rows=2, cols=2, specs = [[{}, {}], [{}, {}]], 
                          subplot_titles=('(1) Top apps in Games genre', '(2) Top apps in Entertainment genre', 
                                          '(3) Top apps in Photo & Video genre','(4) Top apps in Education genre'
                                          ), print_grid=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)

fig['layout']['yaxis1'].update(title='User rating count in Millions')
fig['layout']['yaxis2'].update(title='User rating count in Thousands')
fig['layout']['yaxis3'].update(title='User rating count in Millions')
fig['layout']['yaxis4'].update(title='User rating count in Thousands')
fig['layout'].update(showlegend=False, height=950, )
fig['layout']['margin'].update(l=50, t=50)

py.iplot(fig)
average_cost = df[['prime_genre', 'price']].groupby('prime_genre').price.mean().sort_values(ascending=False).reset_index()

trace = go.Bar(
    x = average_cost['prime_genre'],
    y = average_cost['price'],
    marker = dict(
        color = 'rgb(247, 220, 111)',
        line = dict(
            color = 'rgb(154, 125, 10)',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Average Cost'
    ),
    title = 'Average cost of apps per genre',
    margin = dict(
        l = 100,
        t = 100
    )
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
genre_bs = df[['size_bytes','prime_genre']].groupby(['prime_genre']).size_bytes.mean().sort_values(ascending=False).reset_index()
genre_bs['size_bytes'] = (genre_bs['size_bytes']/1024)/1024 #in MBs

trace = go.Bar(
    x = genre_bs['prime_genre'],
    y = genre_bs['size_bytes'],
    marker = dict(
        color = 'rgb(210, 180, 222)',
        line = dict(
            color = 'rgb(108, 52, 131)',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Average Size in MBs'
    ),
    title = 'Average size of apps per genre',
    margin = dict(
        l = 100,
        t = 100
    )
    
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
genre_ls = df[['prime_genre','lang.num']].groupby(['prime_genre'])['lang.num'].mean().sort_values(ascending=False).reset_index()

trace = go.Bar(
    x = genre_ls['prime_genre'],
    y = genre_ls['lang.num'],
    marker = dict(
        color = 'rgb(171, 178, 185)',
        line = dict(
            color = 'rgb(33, 47, 61)',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Average language support'
    ),
    title = 'Average languages support per genre',
    margin = dict(
        l = 100,
        t = 100
    )
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
genre_ds = df[['prime_genre','sup_devices.num']].groupby(['prime_genre'])['sup_devices.num'].mean().sort_values(ascending=False).reset_index()

trace = go.Bar(
    x = genre_ds['prime_genre'],
    y = genre_ds['sup_devices.num'],
    marker = dict(
        color = 'rgb(162, 217, 206)',
        line = dict(
            color = 'rgb(20, 143, 119)',
            width = 1
        )
    ),
)

data = [trace]

layout = go.Layout(
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Average device support'
    ),
    title = 'Average device support per genre',
    margin = dict(
        l = 100,
        t = 100
    )
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
