import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode (connected = True)
df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')

df.head()
sns.heatmap(df.isna(),cmap='gnuplot')
df['Count']=1

df_titles=df.groupby('type')['Count'].sum().reset_index()
fig1=px.pie(df_titles,values='Count',names='type',hole=0.4)

fig1.update_layout(title='Type of title',title_x=0.5)

fig1.update_traces(textfont_size=15,textinfo='percent+label')

fig1.show()
df_countries=df.groupby('country')['Count'].sum().reset_index().sort_values(by='Count',ascending=False).head(20)
fig2=px.bar(df_countries,x='country',y='Count',color='Count',height=800,width=1000,labels={'country':'Country','Count':'Content produced'})

fig2.update_layout(title='Top 20 producing countries',title_x=0.5)

fig2.show()
df_top20=df[df['country'].isin(['United States', 'India', 'United Kingdom', 'Japan', 'Canada',

       'South Korea', 'Spain', 'France', 'Mexico', 'Turkey', 'Australia',

       'Taiwan', 'Hong Kong', 'United Kingdom, United States', 'Thailand',

       'China', 'Egypt', 'Brazil', 'Philippines', 'Indonesia'])]
fig3=px.sunburst(df_top20,path=['country','type'],names='type')

fig3.update_layout(title="Title type distribution per nation",title_x=0.5,template='plotly_white')

fig3.show()
df['date_added']=df['date_added'].str.strip()

df_months=pd.DataFrame(df['date_added'].str.split(' ').apply(pd.Series).dropna()[0])

df_months.rename(columns={0:'Month released'},inplace=True)

df_months['Count']=1

months_grouped=df_months.groupby('Month released')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)
fig4=px.funnel(months_grouped,x='Count',y='Month released',color='Count',labels={'Count':'Amount of content released'})

fig4.update_layout(title='Popular months of content release',title_x=0.5,template='plotly_dark')

fig4.show()
df_tv=df[df['type']=='TV Show']

df_mov=df[df['type']=='Movie']



df_tv_years=df_tv.groupby('release_year')['Count'].sum().reset_index()

df_mov_years=df_mov.groupby('release_year')['Count'].sum().reset_index()
df_years=df.groupby('release_year')['Count'].sum().reset_index()



fig5=px.line(df_years,x='release_year',y='Count',labels={'release_year':'Year of release','Count':'Amount of content released'})

fig5.update_traces(name='Total content',showlegend=True)

fig5.add_scatter(name='TV shows',x=df_tv_years['release_year'], y=df_tv_years['Count'], mode='lines')

fig5.add_scatter(name='Movies',x=df_mov_years['release_year'], y=df_mov_years['Count'], mode='lines')

fig5.update_layout(title='Number of releases each year',title_x=0.5,template='plotly_dark')

df_ratings=df.dropna().groupby('rating')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

fig6=px.sunburst(df.dropna(),path=['rating','type'],names='type',color='rating',color_discrete_map={'(?)':'black', 'Lunch':'gold', 'Dinner':'darkblue'})

fig6.update_layout(title="Content rating distribution",title_x=0,title_y=0.3,template='plotly_white',margin = dict(t=0, l=0, r=0, b=0))

fig6.show()
df_mov.info()
durations=df_mov['duration'].str.split(' ').apply(pd.Series)
durations=pd.DataFrame(durations[0].astype(int))

durations.rename(columns={0:'Duration'},inplace=True)
fig7=px.histogram(durations,x='Duration',marginal='box',

                  labels={'Duration':'Duration of movie in minutes'},opacity=0.8,color_discrete_sequence=['orange'])

fig7.update_layout(template='plotly_dark',title='Movie duration distribution',title_x=0.5)
from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(20,20))

categories=df['listed_in'].values

wc_cats=WordCloud(max_words=50,background_color='green',collocations=False).generate(str(' '.join(categories)))

plt.imshow(wc_cats)

plt.axis("off")

plt.show()
df_countries_all=df.groupby('country')['Count'].sum().reset_index().sort_values(by='Count',ascending=False).head(126)

map_data = [go.Choropleth( 

           locations = df_countries_all['country'],

           locationmode = 'country names',

           z = df_countries_all["Count"], 

           text = df_countries_all['country'],

           colorbar = {'title':'Amount of content'},

           colorscale='solar')]



layout = dict(title = 'Content per nation', title_x=0.5,

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)