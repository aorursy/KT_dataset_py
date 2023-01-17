# Import the necessary libraries



import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import pycountry

py.init_notebook_mode(connected=True)



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5



# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
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
df = pd.read_csv('/kaggle/input/100-mostfollowed-twitter-accounts-as-of-dec2019/Most_followed_Twitter_acounts_2019.csv')

df.head()
df.info()
df['Followers'] = df['Followers'].str.replace(',', '')

df['Following'] = df['Following'].str.replace(',', '')

df['Tweets'] = df['Tweets'].str.replace(',', '')



df[['Followers','Following','Tweets']] = df[['Followers','Following','Tweets']].astype(int)
# Convert all the names in the same case

df['Name'] = df['Name'].str.title()

df.Name.iloc[:3]
x1 = df['Followers']/1000000

x1 = x1.round(2)



trace = go.Bar(x = df['Followers'][:15],

               y = df['Name'][:15],

               orientation='h',

               marker = dict(color='#00acee',line=dict(color='black')),

               text=x1,

               textposition='auto',

               hovertemplate = "<br>Followers: %{x}")



layout = go.Layout(

                   title='Most followed accounts on Twitter worldwide as of December,2019',

                   width=800, 

                   height=500, 

                   xaxis= dict(title='Number of Followers in Millions'),

                   yaxis=dict(autorange="reversed"),

                   showlegend=False)



fig = go.Figure(data = [trace], layout = layout)



fig.update_layout(width=700,height=500)

fig.show()
df_following = df.sort_values('Following',ascending=False)

following = (df_following['Following']/1000).round(2)



trace = go.Bar(x = df_following['Following'][:15],

               y = df_following['Name'][:15],

               orientation='h',

               marker = dict(color='#00acee',line=dict(color='black',width=0)),

               text=following,

               textposition='auto',

               hovertemplate = "<br>Following: %{x}")



layout = go.Layout(

                   #title='People who follow the most',

                   width=800, 

                   height=500, 

                   xaxis= dict(title='Number of people following in thousands'),

                   yaxis=dict(autorange="reversed"),

                   showlegend=False)



fig = go.Figure(data = [trace], layout = layout)



fig.update_layout(width=700,height=500)

fig.show()
df_following = df.sort_values('Following',ascending=False)

x = df_following['Following'][-15:]

trace = go.Bar(x = df_following['Following'][-15:],

               y = df_following['Name'][-15:],

               orientation='h',

               marker = dict(color='#00acee',line=dict(color='black',width=0)),

               text=x,

               textposition='auto',

               hovertemplate = "<br>Following: %{x}")



layout = go.Layout(

                   #title='People who follow the least',

                   width=800, 

                   height=500, 

                   xaxis= dict(title='Number of people following in thousands'),

                   yaxis=dict(autorange="reversed"),

                   showlegend=False)



fig = go.Figure(data = [trace], layout = layout)



fig.update_layout(width=700,height=500)

fig.show()
counts = df['Nationality/headquarters'].value_counts()

labels = counts.index

values = counts.values



pie = go.Pie(labels=labels, values=values,pull=[0.05, 0], marker=dict(line=dict(color='#000000', width=1)))

layout = go.Layout(title='Region wise Distribution')



fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
df['Industry'].replace({'music':'Music',

                       'news':'News',

                       'sports':'Sports'},inplace=True)









counts = df['Industry'].value_counts()

y= counts.values

trace = go.Bar(x= counts.index,

               y= counts.values,

               marker={'color': y,'colorscale':'Picnic'})



layout = go.Layout(

                   #title='Most followed accounts on Twitter worldwide as of December,2019',

                   width=800, 

                   height=500, 

                   xaxis= dict(title='Industry'),

                   yaxis=dict(title='Count'),

                   showlegend=False)



fig = go.Figure(data = [trace], layout = layout)



fig.update_layout(width=700,height=500)

fig.show()
df_tweets = df.sort_values('Tweets',ascending=False)

tweets = df_tweets['Tweets'][0:10]



trace = go.Bar(x = df_tweets['Tweets'][:10],

               y = df_tweets['Name'][:10],

               orientation='h',

               marker = dict(color='#00acee',line=dict(color='black',width=0)),

               text=tweets,

               textposition='auto',

               hovertemplate = "<br>Tweets: %{x}")



layout = go.Layout(

                   #title='Most active',

                   width=800, 

                   height=500, 

                   xaxis= dict(title='Number of tweets'),

                   yaxis=dict(autorange="reversed"),

                   showlegend=False)



fig = go.Figure(data = [trace], layout = layout)



fig.update_layout(width=700,height=500)

fig.show()
# A generalised barplot and Treemap function

def barplot(data):



    x1 = data['Followers']/1000000

    x1 = x1.round(2)



    trace = go.Bar(x = data['Followers'][:15],

               y = data['Name'][:15],

               orientation='h',

               marker = dict(color='#00acee',line=dict(color='black',width=0)),

               text=x1,

               textposition='auto',

               hovertemplate = "<br>Followers: %{x}")



    layout = go.Layout(

                   #title='Most followed Political accounts on Twitter worldwide as of December,2019',

                   width=800, 

                   height=500, 

                   xaxis= dict(title='Number of Followers in Millions'),

                   yaxis=dict(autorange="reversed"),

                   showlegend=False)



    fig = go.Figure(data = [trace], layout = layout)



    fig.update_layout(width=700,height=500)

    fig.show()
def treemap(data,title):

    import squarify 

    fig = plt.gcf()

    ax = fig.add_subplot()

    fig.set_size_inches(16, 4.5)

    squarify.plot(sizes=data['Activity'].value_counts().values, label=data['Activity'].value_counts().index, 

              alpha=0.5,

              text_kwargs={'fontsize':10,'weight':'bold'},

              color=["red","green","blue", "grey","orange","pink","blue","cyan"])

    plt.axis('off')

    plt.title(title,fontsize=20)

    plt.show()
music = df[df['Industry'] == 'Music']

music['Activity'].replace({'singer-songwriter':'Singer and Songwriter',

                            'Singer and songwriter':'Singer and Songwriter'},inplace=True)





barplot(music)

treemap(music,'Various Categories in Music')
band = df[df['Activity'] == 'Band']

barplot(band)
politics = df[df['Industry'] == 'Politics']



barplot(politics)
sport = df[df['Industry'] == 'Sports']

barplot(sport)
sport['Activity'].replace({'Football league':'Football League'},inplace=True)

treemap(sport,'Various Categories in Sports')

films= df[df['Industry'] == 'Films/Entertainment']

films['Activity'].replace({'Actor ':'Actor'},inplace=True)

barplot(films)

treemap(films,'Various Categories in Films/Entertainment')
tech = df[df['Industry'] =='Technology ']

barplot(tech)

USA=df[df['Nationality/headquarters']=='U.S.A']

barplot(USA)
India=df[df['Nationality/headquarters']=='India']

barplot(India)
UK=df[df['Nationality/headquarters']=='U.K']

barplot(UK)