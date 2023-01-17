import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode (connected = True)
df=pd.read_csv('../input/top-women-chess-players/top_women_chess_players_aug_2020.csv')

df.head()
sns.heatmap(df.isna(),cmap='gnuplot')
unn_cols=['Fide id','Gender']



for cols in unn_cols:

    df.drop(cols,axis=1,inplace=True)
df.dtypes
df['Age']=2020-df['Year_of_birth']
df['Title']=df['Title'].fillna('Unrated')
df['Rapid_rating']=df['Rapid_rating'].fillna(0)

df['Blitz_rating']=df['Blitz_rating'].fillna(0)
df['Inactive_flag']=df['Inactive_flag'].fillna('Active')

df['Inactive_flag']=df['Inactive_flag'].replace('wi','Inactive')
df['Count']=1

df_fed=df.groupby('Federation')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)
fig1=px.bar(df_fed.head(20),x='Federation',y='Count',color='Federation',labels={'Count':'Number of players'})

fig1.update_layout(template='plotly_dark',title="Top 20 most represented nations in women's Chess",title_x=0.5)

fig1.show()
map_data = [go.Choropleth( 

           locations = df_fed['Federation'],

           locationmode = 'ISO-3',

           z = df_fed["Count"], 

           text = df_fed['Federation'],

           colorbar = {'title':'No. of Players'},

           colorscale='cividis')]



layout = dict(title = 'Players per nation', title_x=0.5,

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)
df_title=df.groupby('Title')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

fig2=px.pie(df_title,values='Count',names='Title',hole=0.4)

fig2.update_layout(title='Title distribution of women chess players',title_x=.5,annotations=[dict(text='Title',font_size=20, 

                                                                           showarrow=False,height=800,width=700)])
df_top=df[df['Title'].isin(["GM","IM"])]
fig3=px.sunburst(df_top,path=['Federation','Title'],names='Title')

fig3.update_layout(title="Title distribution per nation",title_x=0.5,template='plotly_white')

fig3.show()
fig4=px.histogram(df,x='Age',marginal='box')

fig4.update_layout(template='plotly_dark',title='Age distribution of women chess players',title_x=0.5)
df_act=df.groupby('Inactive_flag')['Count'].sum().reset_index()

fig5=px.pie(df_act,names='Inactive_flag',values='Count',hole=0.4,color=['Purple','Red'])

fig5.update_layout(title='Activity of women Chess players',title_x=0.5,annotations=[dict(text='Activity',

                                                                                         font_size=15, showarrow=False,

                                                                                         height=800,width=700)])

fig5.update_traces(textfont_size=15,textinfo='percent+label')



fig5.show()
df_a=df[df['Inactive_flag']=='Active']

df_a['Average_rating']=np.round((df_a.iloc[:,4] + df_a.iloc[:,5] + df_a.iloc[:,6])/3,2)
fig6=px.scatter_3d(df_a,x='Standard_Rating',y='Blitz_rating',z='Rapid_rating',

                   color='Average_rating',size='Average_rating',opacity=1,hover_data=['Name','Standard_Rating','Blitz_rating','Rapid_rating','Average_rating'])

fig6.update_layout(margin=dict(l=0, r=0, b=0.5, t=0),title='Active player rating distributions',title_x=0.5,title_y=1)

fig6.update_traces(hovertext='Name')



fig6.show()
df_topGM=df_a.sort_values(by='Average_rating',ascending=False).head(1)

df_topIM=df_a[df_a['Title']=='IM'].sort_values(by='Average_rating',ascending=False).head(1)

df_topFM=df_a[df_a['Title']=='FM'].sort_values(by='Average_rating',ascending=False).head(1)



cats=['Standard rating','Rapid rating','Blitz rating','Average rating']

fig7=go.Figure()

fig7.add_trace(go.Scatterpolar(r=[df_topGM.iloc[0,4],df_topGM.iloc[0,5],df_topGM.iloc[0,6],df_topGM.iloc[0,-1]],

                              theta=cats,fill='toself',name=df_topGM['Name'].values[0]+','+df_topGM['Title'].values[0]))





fig7.add_trace(go.Scatterpolar(r=[df_topIM.iloc[0,4],df_topIM.iloc[0,5],df_topIM.iloc[0,6],df_topIM.iloc[0,-1]],

                              theta=cats,fill='toself',name=df_topIM['Name'].values[0]+','+df_topIM['Title'].values[0]))



fig7.add_trace(go.Scatterpolar(r=[df_topFM.iloc[0,4],df_topFM.iloc[0,5],df_topFM.iloc[0,6],df_topFM.iloc[0,-1]],

                              theta=cats,fill='toself',name=df_topFM['Name'].values[0]+ ','+ df_topFM['Title'].values[0]))



fig7.update_layout(title='Radar plot of ratings of top GM,IM and FM',title_x=0.45)

fig7.show()