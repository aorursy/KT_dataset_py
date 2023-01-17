# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import re

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import plotly.graph_objs as go

import plotly.offline as py

import plotly.figure_factory as ff

import plotly.express as px

from plotly.subplots import make_subplots

import pandas as pd

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df.head()
df.columns
df.info()
#missing data

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
def pie_count(data, field, percent_limit=0.5, title="Plot by "):

    

    title += field

    data[field] = data[field].fillna('NA')

    data = data[field].value_counts().to_frame()



    total = data[field].sum()

    data['percentage'] = 100 * data[field]/total    



    percent_limit = percent_limit

    otherdata = data[data['percentage'] < percent_limit] 

    others = otherdata['percentage'].sum()  

    maindata = data[data['percentage'] >= percent_limit]



    data = maindata

    other_label = "Others(<" + str(percent_limit) + "% each)"           # Create new label

    data.loc[other_label] = pd.Series({field:otherdata[field].sum()}) 

    

    labels = data.index.tolist()   

    datavals = data[field].tolist()

    

    trace=go.Pie(labels=labels,values=datavals)



    layout = go.Layout(

        title = title,

        height=700

        )

    

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)



pie_count(df, 'country')

pie_count(df, 'type')

pie_count(df, 'director',0.1)

tmp = df["country"].value_counts()



# plotly globe credits - https://www.kaggle.com/arthurtok/generation-unemployed-interactive-plotly-visuals

colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 

              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 

              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 

              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]



data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = colorscale,

        showscale = True,

        locations = tmp.index,

        z = tmp.values,

        locationmode = 'country names',

        text = tmp.values,

        marker = dict(

            line = dict(color = '#fff', width = 2)) )           ]



layout = dict(

    height=500,

    title = 'Contents published  by Country',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = '#222',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

iplot(fig)





tmp = df.groupby("country").agg({"show_id" : "sum"}).reset_index()







# plotly globe credits - https://www.kaggle.com/arthurtok/generation-unemployed-interactive-plotly-visuals

colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 

              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 

              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 

              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]



data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = colorscale,

        showscale = True,

        locations = tmp.country,

        z = tmp.show_id,

        locationmode = 'country names',

        text = tmp.show_id,

        marker = dict(

            line = dict(color = '#fff', width = 2)) )           ]



layout = dict(

    height=500,

    title = 'Average content published in each Country',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = '#222',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = False,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )

fig = dict(data=data, layout=layout)

iplot(fig)
df_holly = df[df['country']=='United States']

df_holly["date_added"] = pd.to_datetime(df_holly['date_added'])

df_holly['published_year'] = df_holly['date_added'].dt.year

df_holly['published_month'] = df_holly['date_added'].dt.month

df_holly_movie = df_holly[df_holly['type']=='Movie']

df_holly_tv = df_holly[df_holly['type']=='TV Show']

df_holly['cast'].fillna(value='Actors Not Known',inplace=True)
df_holly_movie['duration'] = pd.to_numeric(df_holly_movie['duration'].str.replace('min',''))
df_holly_movie.head()
df_holly_ = df_holly.copy()
plot_title = df_holly_movie.groupby('title')['duration'].mean().reset_index().sort_values('duration', ascending=True).tail(20)

fig = px.bar(plot_title, x="duration", y="title", orientation='h')

fig.show()
direct = df_holly_movie.groupby('director')['duration'].mean().reset_index().sort_values('duration', ascending=True).tail(20)

fig = px.bar(direct, x="duration", y="director", orientation='h')

fig.show()
Movie_list = {'TV-Y7':'Child Movies',

              'TV-G':'Family Movies',

              'TV-PG':'Family Movies-Parental Guidance',

              'TV-14':'Family Movies-Parental Guidance',

              'TV-MA':'Adult Movies','TV-Y7-FV':'Child Movies',

              'PG-13':'Family Movies-Parental Guidance',

              'PG':'Family Movies-Parental Guidance',

              'R':'Adult Movies',

              'NR':'Unrated Movies',

              'UR':'Unrated Movies'}

df_holly['Movie Type'] = df_holly['rating'].map(Movie_list)



df_holly_movie['Movie Type'] = df_holly_movie['rating'].map(Movie_list)



df_holly_tv['Movie Type'] = df_holly_tv['rating'].map(Movie_list)
col = "Movie Type"



vc1 = df_holly_movie[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)





trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1]

layout = go.Layout(title="Published Content on Netflix for Hollywood", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "published_year"



vc1 = df_holly_movie[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = df_holly_tv[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="blue"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="green"))

data = [trace1, trace2]

layout = go.Layout(title="Year wise content published", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "Movie Type"



vc1 = df_holly_movie[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = df_holly_tv[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="dodgerblue"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="slategrey"))

data = [trace1, trace2]

layout = go.Layout(title="Published content based on Ratings", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
# violin plot 



plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df_holly_movie['published_year'],df_holly_movie['duration'], palette = 'Greys')

plt.title('Comparison of years and Duration of Movies', fontsize = 20)

plt.show()
plt.style.use('dark_background')

df_holly_tv['listed_in'].value_counts().head(80).plot.bar(color = 'red', figsize = (20, 7))

plt.title('Categories published on Netflix (in US)', fontsize = 30, fontweight = 20)

plt.xlabel('Name of The Show Categories')

plt.ylabel('count')

plt.show()
F=df_holly_movie['director'].value_counts().sort_values(ascending=False)[:15]

label=F.index

size=F.values

colors = ['skyblue', '#FEBFB3', '#96D38C', '#D0F9B1', 'gold', 'orange', 'lightgrey', 

          'lightblue','lightgreen','aqua','yellow','#D4E157','#D1C4E9','#1A237E','#64B5F6','#009688',

          '#1DE9B6','#66BB6A','#689F38','#FFB300']

trace =go.Pie(labels=label, values=size, marker=dict(colors=colors), hole=.1)

data_trace = [trace]

layout = go.Layout(title='Top Directors for Hollywood')

fig=go.Figure(data=data_trace,layout=layout)

fig.show()
def category_separator(category,show_id):

    for i in (re.split(r',',category)):

        if i.strip() in df_holly:

            df_holly[i.strip()][df_holly['show_id']==show_id]='YES'

        else:

            df_holly[i.strip()]='NO'

            df_holly[i.strip()][df_holly['show_id']==show_id]='YES'
for show_id, category in zip(df_holly.show_id, df_holly.listed_in): 

    category_separator(category,show_id)
S = df_holly[(df_holly['Anime Series'] == 'YES')]

K = df_holly[(df_holly['Anime Series'] == 'NO')]



trace = go.Bar(y = (len(S),len(K)), x = ['YES','NO'], orientation = 'v',opacity = 0.8, marker=dict(

        color=['green','red'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Distribution for Anime Series')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
df_plot = pd.DataFrame(df_holly['title'][df_holly['Comedies']=='YES'].head(10).reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="count", y="title", barmode='group',orientation='h')

fig.show()
df_plot = pd.DataFrame(df_holly['title'][df_holly['Romantic Movies']=='YES'].head(10).reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="count", y="title", barmode='group',orientation='h')

fig.show()
df_plot = pd.DataFrame(df_holly['title'][(df_holly['Comedies']=='YES')&(df_holly['Movie Type']=='Family Movies-Parental Guidance')].head(15).reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="title", y="count", barmode='group')

fig.show()
def create_stack_bar_data(col):

    aggregated = df[col].value_counts()

    x_values = aggregated.index.tolist()

    y_values = aggregated.values.tolist()

    return x_values, y_values



x1, y1 = create_stack_bar_data('release_year')

x1 = x1[:-1]

y1 = y1[:-1]

trace1 = go.Bar(x=x1, y=y1, opacity=0.75, name="year count", marker=dict(color=['rgba(10, 220, 150, 0.6)', 'rgba(10, 220, 150, 0.6)', 'rgba(10, 220, 150, 0.6)', 'rgba(10, 220, 150, 0.6)', 'rgba(222,45,38,0.8)']))

layout = dict(height=400, title='Contents published in Hollywood by year', legend=dict(orientation="h"));

fig = go.Figure(data=[trace1], layout=layout);

iplot(fig);
dfholly = df_holly[:1000]

def actor_separator(actors,show_id):

    for a in (re.split(r',',actors)):

        if a.strip() in dfholly:

            dfholly[a.strip()][dfholly['show_id']==show_id] = 'YES' 

        else:

            dfholly[a.strip()]='NO'

            dfholly[a.strip()][dfholly['show_id']==show_id]='YES'
for show_id,actors in zip(dfholly['show_id'],dfholly['cast']):

    actor_separator(actors,show_id)
def plot_count(x,fig):

    plt.subplot(4,2,fig)

   

    sns.countplot(y='Movie Type',hue=x,data=dfholly,palette='magma')

    plt.subplot(4,2,(fig+1))

    

    sns.countplot(y='type', hue =x,data = dfholly,palette=("magma"))

    

plt.figure(figsize=(15,20))



plot_count('Robert Downey Jr.', 1)

plot_count('Will Smith', 3)

plot_count('Leonardo DiCaprio',5)

plot_count('Johnny Depp',7)

plt.tight_layout()

plt.show()
def plot_count(x,fig):

    plt.subplot(4,2,fig)

   

    sns.countplot(y='Movie Type',hue=x,data=dfholly,palette='magma')

    plt.subplot(4,2,(fig+1))

    

    sns.countplot(y='type', hue =x,data = dfholly,palette=("magma"))

    

plt.figure(figsize=(15,20))



plot_count('Penélope Cruz', 1)

plot_count('Cameron Diaz', 3)

plot_count('Jennifer Aniston',5)

plot_count('Zhu Zhu',7)

plt.tight_layout()

plt.show()
df_plot = pd.DataFrame(dfholly['title'][(dfholly['Jennifer Aniston']=='YES')&(dfholly['Romantic Movies']=='YES')].reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="count", y="title", barmode='group',orientation='h')

fig.show()
df_plot = pd.DataFrame(dfholly['title'][(dfholly['Bradley Cooper']=='YES')&(dfholly['Comedies']=='YES')].reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="count", y="title", barmode='group',orientation='h')

fig.show()
df_plot = pd.DataFrame(dfholly['title'][(dfholly['Bryan Cranston']=='YES')&(dfholly['Action & Adventure']=='YES')].reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="count", y="title", barmode='group',orientation='h')

fig.show()
df_plot = pd.DataFrame(dfholly['title'][(dfholly['Will Smith']=='YES')&(dfholly['Action & Adventure']=='YES')].reset_index())

df_plot.columns = ['count','title']

fig = px.bar(df_plot, x="count", y="title", barmode='group',orientation='h')

fig.show()