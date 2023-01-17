import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np





#Importando potly

import plotly

import plotly.offline as py



# habilita o modo offline

from plotly.offline import plot

from plotly.offline import iplot

plotly.offline.init_notebook_mode(connected=True)



import plotly.graph_objs as go

import plotly.express as px



import warnings

warnings.filterwarnings('ignore')
filename = "../input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv"

df = pd.read_csv(filename)
df.head()
df.isnull().sum()
df_imdb = df.sort_values(by='IMDb', ascending=False)

print(df_imdb[['Title','Year','IMDb','Age']].head(15))
plt.subplots(figsize=(10,8))

sns.barplot(x="IMDb", y="Title" , data= df_imdb.head(20))
sns.kdeplot(data=df['IMDb'])
imdb = df[['IMDb','Age','Netflix','Hulu','Prime Video', 'Disney+']].groupby(by='IMDb').sum().reset_index()

imdb['total_rate']= imdb[['Netflix','Hulu','Prime Video', 'Disney+']].sum(axis=1)
trace1= go.Scatter(x=imdb['IMDb'],

                y=imdb['total_rate'],

                mode='lines+markers'

                

                )



layout = go.Layout(title='Total Rate IMDB',

                xaxis_rangeslider_visible=True)

fig = go.Figure(trace1,layout)



fig.show()


#IMDB por plataforma

trace1= go.Scatter(x=imdb['IMDb'],

                y=imdb['Netflix'],

                mode='lines+markers',

                name = "Netflix",

                )



trace2= go.Scatter(x=imdb['IMDb'],

                y=imdb['Hulu'],

                mode='lines+markers',

                name = "Hulu",

                )



trace3= go.Scatter(x=imdb['IMDb'],

                y=imdb['Prime Video'],

                mode='lines+markers',

                name = "Prime Video",

                )



trace4= go.Scatter(x=imdb['IMDb'],

                y=imdb['Disney+'],

                mode='lines+markers',

                name = "Disney+",

                )



layout = go.Layout(title='Total Rate IMDB',

                xaxis_rangeslider_visible=True)



data = [trace1,trace2,trace3,trace4]



py.iplot(data)
#Quantidade TV SHOWS por Plataforma



df_sum = df.sum()

df_sum = df_sum['Netflix':'Disney+']

print(df_sum)
trace1 = go.Bar(x=df_sum.values, 

                y=df_sum.index, 

                orientation='h',

                text=df_sum.values,

                textposition='auto'

                

               

               )

x = [trace1]



layout = go.Layout(title="Quantidade TV SHOWS por Plataforma",

                   yaxis={'title':'Plataforma'},

                   xaxis={'title': 'Quantidade'})



fig = go.Figure(x, layout=layout)

fig.show()
df_age = df[['Age','Netflix','Hulu','Prime Video', 'Disney+']].groupby(by='Age').sum().reset_index()

df_age['total_age']= df_age[['Netflix','Hulu','Prime Video', 'Disney+']].sum(axis=1)
trace1= go.Bar(x=df_age['Age'],

               y=df_age['total_age'],

               text=df_age['total_age'],

               textposition='auto'

               )



layout= go.Layout(title='Quantidade Total por Classificação etaria',

                  xaxis={'title':'Classificação etária'},

                  yaxis={'title': 'Quantidade'}

                 

                  )



fig = go.Figure(trace1, layout)

fig.show()
#Faixa etária Netflix

trace1= go.Bar(x=df_age['Age'],

               y=df_age['Netflix'],

               text=df_age['Netflix'],

               textposition='auto')



layout = go.Layout(title='Faixa etária Netflix',

                   xaxis={'title':'Faixa etária'},

                   yaxis={'title':'Quantidade'}

                          )



fig = go.Figure(trace1,layout)

fig.show()
#Faixa etária Hulu

trace1= go.Bar(x=df_age['Age'],

               y=df_age['Hulu'],

               text=df_age['Hulu'],

               textposition='auto')



layout = go.Layout(title='Faixa etária Hulu',

                   xaxis={'title':'Faixa etária'},

                   yaxis={'title':'Quantidade'}

                          )



fig = go.Figure(trace1,layout)

fig.show()
#Faixa etária Hulu

trace1= go.Bar(x=df_age['Age'],

               y=df_age['Prime Video'],

               text=df_age['Prime Video'],

               textposition='auto')



layout = go.Layout(title='Faixa etária Prime Video',

                   xaxis={'title':'Faixa etária'},

                   yaxis={'title':'Quantidade'}

                          )



fig = go.Figure(trace1,layout)

fig.show()
#Faixa etária Disney+

trace1= go.Bar(x=df_age['Age'],

               y=df_age['Disney+'],

               text=df_age['Disney+'],

               textposition='auto')



layout = go.Layout(title='Faixa etária Disney+',

                   xaxis={'title':'Faixa etária'},

                   yaxis={'title':'Quantidade'}

                          )



fig = go.Figure(trace1,layout)

fig.show()
df_year = df[['Year','Age','Netflix','Hulu','Prime Video', 'Disney+']].groupby(by='Year').sum().reset_index()

df_year['total_year']= df_year[['Netflix','Hulu','Prime Video', 'Disney+']].sum(axis=1)
trace1= go.Scatter(x=df_year['Year'],

                y=df_year['total_year'],

                mode='lines+markers'

                

                )



layout = go.Layout(title='Quantidade total por Ano',

                xaxis_rangeslider_visible=True)

fig = go.Figure(trace1,layout)



fig.show()