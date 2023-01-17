import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode (connected = True)
df=pd.read_csv('../input/womens-international-football-results/results.csv')

df.head()
df.dtypes
df['date']=pd.to_datetime(df['date'])
df.isna().any()
df['Year']=df['date'].dt.year

df['Count']=1
df_year=df.groupby('Year')['Count'].sum().reset_index()
fig1=px.line(df_year,y='Count',x='Year',height=600,width=800)

fig1.update_layout(title='Number of matches each year',title_x=0.5,template='plotly_dark')

fig1.update_traces(line_color='#AAF0D1')

fig1.show()

df_h=df.groupby('home_team')['Count'].sum().reset_index().sort_values(by='Count',ascending=False).head(20)

df_a=df.groupby('away_team')['Count'].sum().reset_index().sort_values(by='Count',ascending=False).head(20)
fig2=plt.figure(figsize=(20,15))

ax1=fig2.add_subplot(211)

sns.barplot('home_team','Count',data=df_h,ax=ax1,palette='rocket')

label=df_h['home_team']

ax1.set_xticklabels(label,rotation=90,size=15)

ax1.set_title('Top 20 most active home teams',size=25)

ax1.set_xlabel('Home team',size=15)

ax1.set_ylabel('Number of matches played',size=15)



ax2=fig2.add_subplot(212)

sns.barplot('away_team','Count',data=df_a,ax=ax2,palette='summer')

label=df_a['away_team']

ax2.set_xticklabels(label,rotation=90,size=15)

ax2.set_title('Top 20 most active away teams',size=25)

ax2.set_xlabel('Away team',size=15)

ax2.set_ylabel('Number of matches played',size=15)



fig2.tight_layout(pad=3)
df_tour=df.groupby('tournament')['Count'].sum().reset_index()

fig3=px.pie(df_tour,values='Count',names='tournament',hole=0.3)

fig3.update_layout(title='Tournament wise match distribution',title_x=0.25)

fig3.show()
df['Goal difference']=abs(df['home_score']-df['away_score'])
fig4=px.histogram(df,x='Goal difference',marginal='violin')

fig4.update_layout(title='Goal Difference distribution',title_x=0.5,template='plotly_dark')

fig4.update_traces(opacity=0.9)

fig4.show()
df['Total goals']=df['home_score']+df['away_score']


df_gpg=df.groupby('tournament')[['Count','Total goals']].sum().reset_index()

df_gpg['GPG']=np.round(df_gpg['Total goals']/df_gpg['Count'],0)

df_gpg.sort_values(by='GPG',ascending=False,inplace=True)
fig5=px.bar(df_gpg,x='tournament',y='GPG',color='GPG',height=800,width=1000,labels={'GPG':'Average goals per game','tournament':'Tournament'})

fig5.update_layout(template='plotly_dark')

fig5.show()
df_nn=df[df['neutral']==False]



df_gpg_home=df_nn.groupby('home_team')[['Count','home_score']].sum().reset_index()

df_gpg_home=df_gpg_home[df_gpg_home['Count']>30]

df_gpg_home['GPG']=np.round(df_gpg_home['home_score']/df_gpg_home['Count'],0)

df_gpg_home.sort_values(by='GPG',ascending=False,inplace=True)



df_gpg_away=df_nn.groupby('home_team')[['Count','away_score']].sum().reset_index()

df_gpg_away=df_gpg_away[df_gpg_away['Count']>30]

df_gpg_away['GPG']=np.round(df_gpg_away['away_score']/df_gpg_away['Count'],0)

df_gpg_away.sort_values(by='GPG',ascending=False,inplace=True)
df_gpg_home=df_gpg_home.head(20)

df_gpg_away=df_gpg_away.head(20)




fig6=plt.figure(figsize=(15,15))

ax1=fig6.add_subplot(211)

sns.barplot('home_team','GPG',data=df_gpg_home,palette='gist_earth',ax=ax1)

label1=df_gpg_home['home_team']

ax1.set_xticklabels(label1,rotation=75)

ax1.set_xlabel('Team name',size=15)

ax1.set_ylabel('Average goals scored per game',size=10)

ax1.set_title('Top 20 scoring home teams',size=25)



ax2=fig6.add_subplot(212)

sns.barplot('home_team','GPG',data=df_gpg_away,palette='coolwarm',ax=ax2)

label2=df_gpg_away['home_team']

ax2.set_xticklabels(label2,rotation=75)

ax2.set_xlabel('Team name',size=15)

ax2.set_ylabel('Average goals scored per game',size=10)

ax2.set_title('Top 20 scoring away teams',size=25)



fig6.tight_layout(pad=3)
from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize=(20,20))

text_city=df['city'].values

wc_city=WordCloud(max_words=50,background_color='white',collocations=False).generate(str(' '.join(text_city)))

plt.imshow(wc_city)

plt.axis("off")

plt.show()

plt.figure(figsize=(20,20))

text_country=df['country'].values

wc_country=WordCloud(max_words=50

                     ,background_color='green',collocations=False).generate(str(' '.join(text_country)))

plt.imshow(wc_country)

plt.axis("off")

plt.show()
df_venue=df.groupby('neutral')['Count'].sum().reset_index()
fig7=px.pie(df_venue,values='Count',names='neutral',hole=0.4,color_discrete_sequence=['orange','green'])

fig7.update_layout(title='Type of venue',title_x=.5,annotations=[dict(text='Neutral ground',font_size=15, showarrow=False,height=800,width=700)])

fig7.update_traces(textfont_size=15,textinfo='percent+label')





fig7.show()