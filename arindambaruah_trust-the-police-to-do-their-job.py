import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots
df=pd.read_csv('../input/us-police-shootings/shootings.csv')

df.head()
df.drop('id',axis=1,inplace=True)

df.drop('name',axis=1,inplace=True)
df['Count']=1

df.isna().any()
df.info()
sns.catplot('manner_of_death',data=df,kind='count',palette='summer',height=5,aspect=2)

plt.xlabel('Manner of death',size=15)

plt.ylabel('Number of cases',size=15)

plt.title('Classification of death',size=20)
fig1=plt.figure(figsize=(10,8))

ax1=fig1.add_subplot(121)

a=sns.distplot(df['age'])

plt.title('Age distribution of offenders shot',size=15)

a.axvline(df['age'].median(),color='red',label='Median age')

ax1.set_xlabel('Age',size=15)

ax1.legend()



ax2=fig1.add_subplot(122)

sns.violinplot(df['age'],inner='quartile',palette='summer',orient='v')

ax2.set_title('Median age:{}'.format(df['age'].median()),size=15)
df_gend=df.groupby('gender')['Count'].sum().reset_index()

plt.figure(figsize=(10,8))

labels=['Female','Male']

plt.pie(df_gend['Count'],autopct="%1.1f%%",explode=(0.2,0),labels=labels)

plt.title('Gender distribution of the shootings',size=20)


df_race=df.groupby('race')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_race



fig2=go.Figure([go.Pie(labels=df_race['race'],values=df_race['Count'])])



fig2.update_traces(textfont_size=15,textinfo='value+percent')

fig2.update_layout(title='Shootings based on races',title_x=0.5,height=700,width=700)

fig2.show()
fig3=plt.figure(figsize=(15,8))

ax1=fig3.add_subplot(121)

ax1.set_title('Violinplot of ages based on race of offenders',size=15)

sns.violinplot(df['race'],df['age'],inner='quartile',palette='coolwarm',ax=ax1)

ax1.set_xlabel('Race',size=15)

ax1.set_ylabel('Age',size=15)

ax2=fig3.add_subplot(122)

sns.boxplot(df['race'],df['age'],palette='summer',ax=ax2)

ax2.set_title('Boxplot of ages based on race of offenders',size=15)

ax2.set_xlabel('Race',size=15)

ax2.set_ylabel('Age',size=15)
df_mental=df[df['signs_of_mental_illness']==True]

df_grouped=df_mental.groupby('race')['Count'].sum().reset_index()
df_grouped
fig4=go.Figure([go.Pie(labels=df_grouped['race'],values=df_grouped['Count'])])



fig4.update_traces(textfont_size=15,textinfo='value+percent')

fig4.update_layout(title='Mentally sick offenders based on race',title_x=0.5,height=700,width=700)

fig4.show()
sns.catplot('race',kind='count',data=df,hue='threat_level',palette='viridis',height=8,aspect=2)

plt.xticks(size=15)

plt.xlabel('Race',size=20)

plt.yticks(size=15)

plt.ylabel('Count',size=20)

plt.title('Threat level of each of the races',size=25)

plt.legend(fontsize=15)

df_top10=df.groupby('city')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_top10=df_top10.head(10)

df_top10
sns.catplot('city','Count',data=df_top10,kind='bar',aspect=2,height=8,palette='summer')

plt.xlabel('City',size=20)

plt.ylabel('Number of shootings',size=20)

plt.title('Top 10 cities with highest shootings',size=25)

plt.xticks(size=15)
df_black=df[df['race']=='Black']

df_black_grouped=pd.DataFrame(df_black['state'].value_counts())

df_black_grouped.reset_index(inplace=True)

df_black_grouped.rename(columns={'index':'state','state':'count'},inplace=True)






fig = go.Figure(go.Choropleth(

    locations=df_black_grouped['state'],

    z=df_black_grouped['count'].astype(float),

    locationmode='USA-states',

    colorscale='Reds',

    autocolorscale=False,

    text=df_black_grouped['state'], # hover text

    marker_line_color='white', # line markers between states

    colorbar_title="Millions USD",showscale = False,

))

fig.update_layout(

    title_text='States with high black police shootings',

    title_x=0.5,

    geo = dict(

        scope='usa',

        projection=go.layout.geo.Projection(type = 'albers usa'),

        showlakes=True, # lakes

        lakecolor='rgb(255, 255, 255)'))



fig.show()