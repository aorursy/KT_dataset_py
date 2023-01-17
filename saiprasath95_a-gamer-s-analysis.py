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
pd.set_option('display.max_columns', None)  

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline 

import seaborn as sns

import plotly.io as pio

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("/kaggle/input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv")

df.head()
df.isnull().sum()
df = df[df["Year_of_Release"].notnull()]

df = df[df["Genre"].notnull()]

df = df[df["Publisher"].notnull()]

df['Year_of_Release']=df['Year_of_Release'].astype('int64')

df['User_Score']=df['User_Score'].replace('tbd',0).astype('float64')
sns.set_style("whitegrid")

trace1=go.Scatter(

                x=df.groupby(['Genre']).mean().reset_index()['Genre'], 

                y=df.groupby(['Genre']).mean().reset_index()['NA_Sales'],

                mode='lines+markers',

                name='North America Sales',

                marker = dict(size=8),

                line=dict(color = '#FA8072',width=2.5))

trace2=go.Scatter(

                x=df.groupby(['Genre']).mean().reset_index()['Genre'], 

                y=df.groupby(['Genre']).mean().reset_index()['EU_Sales'],

                mode='lines+markers',

                name='Europe Sales',

                marker = dict(size=8),

                line=dict(color = '#6495ED',width=2.5))



trace3=go.Scatter(

                x=df.groupby(['Genre']).mean().reset_index()['Genre'], 

                y=df.groupby(['Genre']).mean().reset_index()['JP_Sales'],

                mode='lines+markers',

                name='Japan Sales',

                marker = dict(size=8),

                line=dict(color = 'yellowgreen',width=2.5))



trace4=go.Scatter(

                x=df.groupby(['Genre']).mean().reset_index()['Genre'], 

                y=df.groupby(['Genre']).mean().reset_index()['Other_Sales'],

                mode='lines+markers',

                name='Other Country Sales',

                marker = dict(size=8),

                line=dict(color = '#DAA520',width=2.5))



edit_df=[trace1,trace2,trace3,trace4]

layout=dict(

            legend=dict(x=0.77, y=1.2, font=dict(size=10)), legend_orientation="v",

            title="Average Sales of Different Genre Games",

            xaxis=dict(title="Genre",tickfont=dict(size=8.35),zeroline=False,gridcolor="white"),

            yaxis=dict(title='Average Sales in Different Countries',gridcolor="#DCDCDC"),

            plot_bgcolor='white')





fig=dict(data=edit_df,layout=layout)

iplot(fig)
fig=plt.figure(figsize=(29,14))

plt.subplots_adjust(left=0.25, wspace=0.20, hspace=0.35)

sns.set_style("white")



plt.subplot(2, 2, 1)

plt.title('Gross Sales of Different Genre Games in Europe',fontdict={'fontsize':16})

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.barplot(y='Genre', x='EU_Sales', data=df.groupby('Genre').sum().EU_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');

plt.ylabel('Genre',fontdict={'fontsize':16})

plt.xlabel('Sales in Europe',fontdict={'fontsize':16})



plt.subplot(2, 2, 2)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Gross Sales of Different Genre Games in North America',fontdict={'fontsize':16})

sns.barplot(y='Genre', x='NA_Sales', data=df.groupby('Genre').sum().NA_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');

plt.ylabel('',fontdict={'fontsize':16})

plt.xlabel('Sales in North America',fontdict={'fontsize':16})



plt.subplot(2, 2, 3)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Gross Sales of Different Genre Games in Japan',fontdict={'fontsize':16})

sns.barplot(y='Genre', x='JP_Sales', data=df.groupby('Genre').sum().JP_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');

plt.ylabel('Genre',fontdict={'fontsize':16})

plt.xlabel('Sales in Japan',fontdict={'fontsize':16})





plt.subplot(2, 2, 4)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Gross Sales of Different Genre Games in Other Countries',fontdict={'fontsize':16})

sns.barplot(y='Genre', x='Other_Sales', data=df.groupby('Genre').sum().Other_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');

plt.ylabel('',fontdict={'fontsize':16})

plt.xlabel('Sales in Other Countries',fontdict={'fontsize':16})



fig=plt.figure(figsize=(24.5,22))

plt.subplot2grid((3,1), (1,0))

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Gross Global Sales of Different Genre Games',fontdict={'fontsize':16})

sns.barplot(y='Genre', x='Global_Sales', data=df.groupby('Genre').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='YlOrRd_r');

plt.ylabel('Genre',fontdict={'fontsize':16})

plt.xlabel('Global Sales',fontdict={'fontsize':16});
fig=plt.figure(figsize=(24.5,22))

sns.set_style("white")

plt.subplot2grid((3,1), (1,0))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Total Number of Games released in Each Genre',fontdict={'fontsize':16})

sns.barplot(y=df['Genre'].value_counts().index,x=df['Genre'].value_counts(),palette='YlOrRd_r')

plt.ylabel('Genre',fontdict={'fontsize':16})

plt.xlabel('Number of Games',fontdict={'fontsize':16});
fig=plt.figure(figsize=(24,10))

plt.subplots_adjust(left=None, wspace=None, hspace=None)

sns.set_style("whitegrid")



plt.subplot(1, 2, 1)

plt.title('Global Sales over Years',fontdict={'fontsize':14})

plt.xticks(rotation=90)

sns.barplot(x='Year_of_Release',y='Global_Sales',data=df.groupby(df['Year_of_Release'].sort_values()).sum().Global_Sales.reset_index(),palette='plasma')

plt.ylabel('Global Sales',fontdict={'fontsize':13})

plt.xlabel('Year of Release',fontdict={'fontsize':13})



plt.subplot(1, 2, 2)

plt.title('Number of Games released every Year',fontdict={'fontsize':14})

plt.xticks(rotation=90)

sns.barplot(x=df.Year_of_Release.value_counts().index, y=df.Year_of_Release.value_counts(),palette='plasma');

plt.ylabel('Number of Games',fontdict={'fontsize':13})

plt.xlabel('Year of Release',fontdict={'fontsize':13});
sc = StandardScaler()

Year_Count_Sales=df.groupby(df['Year_of_Release']).apply(lambda x: pd.Series({

    'Count'       : x['Name'].count(),

    'Global_Sales'       : x['Global_Sales'].sum()})).reset_index()



Year_Count_Sales_Scaled = pd.concat([Year_Count_Sales['Year_of_Release'],pd.DataFrame(sc.fit_transform(Year_Count_Sales[['Count','Global_Sales']]),columns=['Count', 'Global_Sales'])],axis=1)



fig = go.Figure(data=[

    go.Scatter(

                x=Year_Count_Sales_Scaled['Year_of_Release'], 

                y=Year_Count_Sales_Scaled['Count'],

                mode='lines+markers',

                name='Number of Games Released',

                marker = dict(size=8),

                line=dict(color = '#FA8072',width=2.5),

                text=Year_Count_Sales['Count'],

                

                hovertemplate = '<i>Year</i>: %{x}'

                                '<br><i>Number of Games</i>: %{text}<br>'),

    go.Scatter(

                x=Year_Count_Sales_Scaled['Year_of_Release'], 

                y=Year_Count_Sales_Scaled['Global_Sales'],

                mode='lines+markers',

                name='Global Sales',

                marker = dict(size=8),

                line=dict(color = '#6495ED',width=2.5),

                text = Year_Count_Sales['Global_Sales'],

                hovertemplate = '<i>Year</i>: %{x}'

                                '<br><i>Global_Sales</i>: %{text}<br>')



],layout=dict(legend=dict(x=0.73, y=1.15, font=dict(size=10)),legend_orientation="v",title="Relationship between Number of Releases and Global Sales",

            xaxis=dict(tickmode = 'linear',tickangle=-90,tickfont=dict(size=10),title="Year of Release",tickwidth=5,ticklen=8,zeroline=True,gridcolor="white",

             showline=True),

            yaxis=dict(title="Number of Release / Global Sales",zeroline=True,showline=True,gridcolor="#DCDCDC",

                         showgrid=True,

        zerolinecolor='#DCDCDC',

        zerolinewidth=1)

            ,plot_bgcolor='white'))



fig.show()
Yr_Sls=df.iloc[:, [2,0,9]].groupby(['Year_of_Release']).Global_Sales.max().reset_index()

Gm_yr_Sls=df[['Name','Global_Sales','Year_of_Release']]

Gm_of_Yr=pd.merge(Yr_Sls, Gm_yr_Sls,on=['Year_of_Release','Global_Sales'],how='left')



trace = go.Bar(x=Gm_of_Yr.Year_of_Release,

               y=Gm_of_Yr.Global_Sales,

               marker=dict(color=Gm_of_Yr.Global_Sales,colorscale='emrld'),

               opacity=0.90,

               name='Game of the Year',

               text = Gm_of_Yr['Name'],

               hovertemplate = '<i>Year: %{x}</i>'

                               '<br><i>Game: %{text}</i>'

                               '<br><i>Global Sales: %{y}</i>')



layout = go.Layout(

    title='Top Selling Game of the Year',

    xaxis=dict(tickmode = 'linear',tickfont=dict(size=11),

        title='Year of Release',tickwidth=5,ticklen=8,zeroline=False,

    tickangle=-90),

    yaxis=dict(

        title='Global Sales'),

    bargap=0.2,

    bargroupgap=0.1, 

    plot_bgcolor="white")

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
publ_sls=df.groupby(['Year_of_Release','Publisher']).Global_Sales.sum().reset_index()

publ_yr_sls=publ_sls.groupby('Year_of_Release').Global_Sales.max().reset_index()

Pub_of_Yr=pd.merge(publ_yr_sls,publ_sls,on=['Year_of_Release','Global_Sales'],how='left')



trace = go.Bar(x=Pub_of_Yr.Year_of_Release,

               y=Pub_of_Yr.Global_Sales,

               marker=dict(color=Pub_of_Yr.Global_Sales,colorscale='burgyl'),

               opacity=0.90,

               name='Publisher of the Year',

               text = Pub_of_Yr['Publisher'],

               hovertemplate = '<i>Year: %{x}</i>'

                               '<br><i>Publisher: %{text}</i>'

                               '<br><i>Global Sales: %{y}</i>')

layout = go.Layout(

    title='Publiser of the Year with Highest Global Sales',

    xaxis=dict(tickmode = 'linear',tickfont=dict(size=11),

        title='Year of Release',tickwidth=5,ticklen=8,zeroline=False,

    tickangle=-90),

    yaxis=dict(tick0=0,

        dtick=25,

        title='Global Sales'

    ),

    bargap=0.2,

    bargroupgap=0.1, 

    plot_bgcolor="white")

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
print("Publiser who made the highest sales most of the years:",Pub_of_Yr['Publisher'].mode()[0])
go.Figure(data=[

    go.Scatter(

                x=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['Year_of_Release'], 

                y=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['NA_Sales'],

                mode='lines+markers',

                name='North America Sales',

                marker = dict(size=8),

                line=dict(color = '#FA8072',width=2.5),

                text = Pub_of_Yr['Publisher'],

                hovertemplate = '<i>Year: %{x}</i>'

                               '<br><i>Sales: %{y} </i>'),

    go.Scatter(

                x=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['Year_of_Release'], 

                y=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['EU_Sales'],

                mode='lines+markers',

                name='Europe Sales',

                marker = dict(size=8),

                line=dict(color = '#6495ED',width=2.5),

                text = Pub_of_Yr['Publisher'],

                hovertemplate = 'Year: %{x}'

                               '<br><i>Sales: %{y} </i>'),

    go.Scatter(

                x=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['Year_of_Release'], 

                y=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['JP_Sales'],

                mode='lines+markers',

                name='Japan Sales',

                marker = dict(size=8),

                line=dict(color = 'yellowgreen',width=2.5),

                text = Pub_of_Yr['Publisher'],

                hovertemplate = '<i>Year: %{x}</i>'

                               '<br><i>Sales: %{y} </i>'),

    go.Scatter(

                x=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['Year_of_Release'], 

                y=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index()['Other_Sales'],

                mode='lines+markers',

                name='Other Country Sales',

                marker = dict(size=8),

                line=dict(color = '#DAA520',width=2.5),

                text = Pub_of_Yr['Publisher'],

                hovertemplate = '<i>Year: %{x}</i>'

                               '<br><i>Sales: %{y} </i>')

    

],layout=dict(legend=dict(x=-0.04, y=1.09,font=dict(size=10)),legend_orientation="h",title="Sales of Games in Different Countries over the Years",

            xaxis=dict(tickmode = 'linear',tickangle=-90,tickfont=dict(size=10),title="Year of Release",tickwidth=5,ticklen=8,zeroline=False,gridcolor="white"),

            yaxis=dict(title="Gross Sales in Different Countries",gridcolor="#DCDCDC")

            ,paper_bgcolor='white',plot_bgcolor='white'))


sns.set_style("whitegrid")

fig=plt.figure(figsize=(23.5,10))

plt.xticks(rotation=90)

colo='yellowgreen'

plt.title('Sales of Games in Different Countries over the Years',fontdict={'fontsize':14})

for i in range(0,len(df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum().sort_values(ascending=False).index)):

    sns.pointplot(x='Year_of_Release',y=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum().sort_values(ascending=False).index[i],data=df.groupby(df['Year_of_Release'].sort_values()).sum().drop(['Year_of_Release'],axis=1).reset_index(),color=colo)

    i=i+1

    colo='grey'



plt.legend(handles=[mpatches.Patch(color='yellowgreen', label=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum().sort_values(ascending=False).index[0]),

                       mpatches.Patch(color='grey', label=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum().sort_values(ascending=False).index[1]),

                       mpatches.Patch(color='grey', label=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum().sort_values(ascending=False).index[2]),

                       mpatches.Patch(color='grey', label=df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum().sort_values(ascending=False).index[3])], loc='upper left', fontsize = 12)    



plt.ylabel('Gross Sales in Different Countries',fontdict={'fontsize':13})

plt.xlabel('Year of Release',fontdict={'fontsize':13});
fig=plt.figure(figsize=(24.5,8.5))

# plt.subplots_adjust(left=None, wspace=None, hspace=None)

sns.set_style("white")



plt.subplot(1, 2, 1)

plt.xticks(rotation=90)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Global Sales of Different Platform Games',fontdict={'fontsize':16})

Plat_sales=df.groupby(df['Platform']).sum().Global_Sales.sort_values(ascending=False).reset_index()

sns.barplot(x='Platform',y='Global_Sales',data=Plat_sales,palette='summer_r');

plt.ylabel('Global Sales',fontdict={'fontsize':16})

plt.xlabel('Platform',fontdict={'fontsize':16})



plt.subplot(1, 2, 2)

plt.xticks(rotation=90)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Number of Games in Each Platform',fontdict={'fontsize':16})

sns.barplot(x=df.Platform.value_counts()[df.Platform.value_counts()>10].index, y=df.Platform.value_counts()[df.Platform.value_counts()>10],palette='summer_r');

plt.ylabel('Number of Games',fontdict={'fontsize':16})

plt.xlabel('Platform',fontdict={'fontsize':16});

sc = StandardScaler()

Plat_Count_Sales=df.groupby(df['Platform']).apply(lambda x: pd.Series({

    'Count'       : x['Name'].count(),

    'Global_Sales'       : x['Global_Sales'].sum()})).reset_index()



Plat_Count_Sales_Scaled=pd.concat([Plat_Count_Sales['Platform'],pd.DataFrame(sc.fit_transform(Plat_Count_Sales[['Count','Global_Sales']]),columns=['Count', 'Global_Sales'])],axis=1)



fig = go.Figure(data=[

    go.Scatter(

                x=Plat_Count_Sales_Scaled['Platform'], 

                y=Plat_Count_Sales_Scaled['Count'],

                mode='lines+markers',

                name='Number of Games Released',

                marker = dict(size=8),

                line=dict(color = 'yellowgreen',width=2.5),

                text=Plat_Count_Sales['Count'],

                

                hovertemplate = '<i>Platform</i>: %{x}'

                                '<br><i>Number of Games</i>: %{text}<br>'),

    go.Scatter(

                x=Plat_Count_Sales_Scaled['Platform'], 

                y=Plat_Count_Sales_Scaled['Global_Sales'],

                mode='lines+markers',

                name='Global Sales',

                marker = dict(size=8),

                line=dict(color = '#6495ED',width=2.5),

                text = Plat_Count_Sales['Global_Sales'],

                hovertemplate = '<i>Platform</i>: %{x}'

                                '<br><i>Global_Sales</i>: %{text}<br>')



],layout=dict(legend=dict(x=-0.02, y=1.11,font=dict(size=10)),legend_orientation="h",title="Relationship between Number of Releases and Global Sales of Different Plaform Games",

            xaxis=dict(tickmode = 'linear',tickangle=-90,tickfont=dict(size=10),title="Platform",tickwidth=5,ticklen=8,zeroline=True,gridcolor="white",

             showline=True),

            yaxis=dict(tick0=-1,dtick=1,title="Number of Release / Global Sales",showticklabels=True,gridcolor="#DCDCDC",

                         showgrid=True,

        zerolinecolor='#DCDCDC',

        zerolinewidth=1)

            ,plot_bgcolor='white'))



fig.show()
fig=plt.figure(figsize=(24.5,16.5))

sns.set_style("white")



plt.subplot(2, 2, 1)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Sales of Different Platform Games in North America',fontdict={'fontsize':16})

Plat_salesNA=df.groupby(df['Platform']).sum().NA_Sales.sort_values(ascending=False).reset_index()

sns.barplot(y='Platform',x='NA_Sales',data=Plat_salesNA[Plat_salesNA['NA_Sales']>10],palette='summer_r')

plt.ylabel('Platform',fontdict={'fontsize':16})

plt.xlabel('Sales in North America',fontdict={'fontsize':16});



plt.subplot(2, 2, 2)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Sales of Different Platform Games in Europe',fontdict={'fontsize':16})

Plat_salesEU=df.groupby(df['Platform']).sum().EU_Sales.sort_values(ascending=False).reset_index()

sns.barplot(y='Platform',x='EU_Sales',data=Plat_salesEU[Plat_salesEU['EU_Sales']>=10],palette='summer_r')

plt.ylabel('Platform',fontdict={'fontsize':16})

plt.xlabel('Sales in Europe',fontdict={'fontsize':16});



plt.subplot(2, 2, 3)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Sales of Different Platform Games in Japan',fontdict={'fontsize':16})

Plat_salesJP=df.groupby(df['Platform']).sum().JP_Sales.sort_values(ascending=False).reset_index()

sns.barplot(y='Platform',x='JP_Sales',data=Plat_salesJP[Plat_salesJP['JP_Sales']>=10],palette='summer_r')

plt.ylabel('Platform',fontdict={'fontsize':16})

plt.xlabel('Sales in Japan',fontdict={'fontsize':16});



plt.subplot(2, 2, 4)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Sales of Different Platform Games in Other Countries',fontdict={'fontsize':16})

Plat_salesOT=df.groupby(df['Platform']).sum().Other_Sales.sort_values(ascending=False).reset_index()

sns.barplot(y='Platform',x='Other_Sales',data=Plat_salesOT[Plat_salesOT['Other_Sales']>=1],palette='summer_r')

plt.ylabel('Platform',fontdict={'fontsize':16})

plt.xlabel('Sales in Other Countries',fontdict={'fontsize':16});
fig=plt.figure(figsize=(24.5,7))



sns.set_style("white")



plt.subplot(1, 2, 1)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xticks(rotation=0)

Plat_US=df.groupby(df['Platform']).mean().User_Score.sort_values(ascending=False).reset_index()

plt.title('Average User Scores of Different Platform Games',fontdict={'fontsize':16})

sns.barplot(y='Platform',x='User_Score',data=Plat_US[Plat_US['User_Score'].notnull()],palette='summer_r')

plt.ylabel('Platform',fontdict={'fontsize':16})

plt.xlabel('User Scores',fontdict={'fontsize':16});



plt.subplot(1, 2, 2)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xticks(rotation=0)

Plat_CS=df.groupby(df['Platform']).mean().Critic_Score.sort_values(ascending=False).reset_index()

plt.title('Average Critic Scores of Different Platform Games',fontdict={'fontsize':16})

sns.barplot(y='Platform',x='Critic_Score',data=Plat_CS[Plat_CS['Critic_Score'].notnull()],palette='summer_r')

plt.ylabel('Platform',fontdict={'fontsize':16})

plt.xlabel('Critic Scores',fontdict={'fontsize':16});
fig=plt.figure(figsize=(20,9))

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Top Game Publishers',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x=df['Publisher'].value_counts()[df['Publisher'].value_counts()>50].index,y=df['Publisher'].value_counts()[df['Publisher'].value_counts()>50],palette='cool')

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('Number of Releases',fontdict={'fontsize':16});
fig=plt.figure(figsize=(20,14))

plt.subplots_adjust(left=None, wspace=0.50, hspace=0.30)

sns.set_style("white")



plt.subplot(2, 2, 1)

plt.xticks(fontsize=10.5)

plt.yticks(fontsize=10.5)

plt.title('Top Publishers in Europe',fontdict={'fontsize':16})

sns.barplot(y='Publisher', x='EU_Sales', data=df.groupby('Publisher').sum().EU_Sales.sort_values(ascending=False).reset_index().head(20),palette='cool_r');

plt.ylabel('Publisher',fontdict={'fontsize':16})

plt.xlabel('Sales in Europe',fontdict={'fontsize':16});



plt.subplot(2, 2, 2)

plt.xticks(fontsize=10.5)

plt.yticks(fontsize=10.5)

plt.title('Top Publishers in North America',fontdict={'fontsize':16})

sns.barplot(y='Publisher', x='NA_Sales', data=df.groupby('Publisher').sum().NA_Sales.sort_values(ascending=False).reset_index().head(20),palette='cool_r');

plt.ylabel('',fontdict={'fontsize':16})

plt.xlabel('Sales in North America',fontdict={'fontsize':16});



plt.subplot(2, 2, 3)

plt.xticks(fontsize=10.5)

plt.yticks(fontsize=10.5)

plt.title('Top Publishers in Japan',fontdict={'fontsize':16})

sns.barplot(y='Publisher', x='JP_Sales', data=df.groupby('Publisher').sum().JP_Sales.sort_values(ascending=False).reset_index().head(20),palette='cool_r');

plt.ylabel('Publisher',fontdict={'fontsize':16})

plt.xlabel('Sales in Japan',fontdict={'fontsize':16});



plt.subplot(2, 2, 4)

plt.xticks(fontsize=10.5)

plt.yticks(fontsize=10.5)

plt.title('Top Publishers in Other Countries',fontdict={'fontsize':16})

sns.barplot(y='Publisher', x='Other_Sales', data=df.groupby('Publisher').sum().Other_Sales.sort_values(ascending=False).reset_index().head(20),palette='cool_r');

plt.ylabel('',fontdict={'fontsize':16})

plt.xlabel('Sales in Other Countries',fontdict={'fontsize':16});



fig=plt.figure(figsize=(20,22))

plt.subplot2grid((3,1), (1,0))

plt.xticks(fontsize=11)

plt.yticks(fontsize=11)

plt.title('Top Publishers Globally',fontdict={'fontsize':16})

sns.barplot(y='Publisher', x='Global_Sales', data=df.groupby('Publisher').sum().Global_Sales.sort_values(ascending=False).reset_index().head(20),palette='cool_r');

plt.ylabel('Publisher',fontdict={'fontsize':16})

plt.xlabel('Global Sales',fontdict={'fontsize':16});
fig=plt.figure(figsize=(23,7))

plt.subplots_adjust(left=None, wspace=0.20, hspace=None)

sns.set_style("white")



plt.subplot(1, 2, 1)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Top Publishers with Highest User Score',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x='Publisher', y='User_Score', data=df.groupby('Publisher').sum().sort_values(by='User_Score',ascending=False).reset_index().head(20),palette='cool_r');

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('User Score',fontdict={'fontsize':16});



plt.subplot(1, 2, 2)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Top Publishers with Higest Critic Score',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x='Publisher', y='Critic_Score', data=df.groupby('Publisher').sum().sort_values(by='Critic_Score',ascending=False).reset_index().head(20),palette='cool_r');

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('Critic Score',fontdict={'fontsize':16});



fig=plt.figure(figsize=(20,23))

plt.subplot2grid((3,1), (1,0))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Top Games with Highest Global Sales',fontdict={'fontsize':16})

sns.barplot(y='Name',x='Global_Sales',data=df.sort_values(by='Global_Sales',ascending=False).head(15),palette='RdPu_r')

plt.xlabel('Global Sales',fontdict={'fontsize':16})

plt.ylabel('Game',fontdict={'fontsize':16});
fig=plt.figure(figsize=(18.5,12))

plt.subplots_adjust(left=None, wspace=0.60, hspace=0.35)



plt.subplot(2, 2, 1)

plt.title('Top Games with Higest Sales in North America',fontdict={'fontsize':16})

sns.barplot(y='Name',x='NA_Sales',data=df.sort_values(by='NA_Sales',ascending=False).head(14),palette='RdPu_r')

plt.xlabel('Sales in North America',fontdict={'fontsize':16})

plt.ylabel('Game',fontdict={'fontsize':16});



plt.subplot(2, 2, 2)

plt.title('Top Games with Higest Sales in Europe',fontdict={'fontsize':16})

sns.barplot(y='Name',x='EU_Sales',data=df.sort_values(by='EU_Sales',ascending=False).head(14),palette='RdPu_r')

plt.xlabel('Sales in Europe',fontdict={'fontsize':16})

plt.ylabel('',fontdict={'fontsize':16});



plt.subplot(2, 2, 3)

plt.title('Top Games with Higest Sales in Japan',fontdict={'fontsize':16})

sns.barplot(y='Name',x='JP_Sales',data=df.sort_values(by='JP_Sales',ascending=False).head(14),palette='RdPu_r')

plt.xlabel('Sales in Japan',fontdict={'fontsize':16})

plt.ylabel('Game',fontdict={'fontsize':16});



plt.subplot(2, 2, 4)

plt.title('Top Games with Higest Sales in Other Countries',fontdict={'fontsize':16})

sns.barplot(y='Name',x='Other_Sales',data=df.sort_values(by='Other_Sales',ascending=False).head(14),palette='RdPu_r')

plt.xlabel('Sales in Other Countries',fontdict={'fontsize':16})

plt.ylabel('',fontdict={'fontsize':16});

fig=plt.figure(figsize=(14,6))

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Top Games with Higest User Score',fontdict={'fontsize':16})

sns.barplot(y='Name',x='User_Score',data=df.sort_values(by=['User_Score','User_Count'],ascending=False).head(16),palette='RdPu_r')

plt.xlabel('User Score',fontdict={'fontsize':16})

plt.ylabel('Game',fontdict={'fontsize':16});
xrange=np.arange(0,110,5)

fig=plt.figure(figsize=(14,6))

sns.set_style("white")

plt.xticks(xrange)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Top Games with Higest Critic Score',fontdict={'fontsize':16})

sns.barplot(y='Name',x='Critic_Score',data=df.sort_values(by=['Critic_Score','Critic_Count'],ascending=False).head(20),palette='RdPu_r',ci=None)

plt.xlabel('Critic Score',fontdict={'fontsize':16})

plt.ylabel('Game',fontdict={'fontsize':16});
df20=df[df['Year_of_Release']<=2000]

df21=df[df['Year_of_Release']>2000]
index = ['North America', 'Europe', 'Japan', 'Other Country', 'Global']

  

# Convert the dictionary into DataFrame  

# Make Own Index and Removing Default index 

salcent=pd.DataFrame({'20th Century':[df20['NA_Sales'].sum(),df20['EU_Sales'].sum(),df20['JP_Sales'].sum(),df20['Other_Sales'].sum(),df20['Global_Sales'].sum()],'21st Century':[df21['NA_Sales'].sum(),df21['EU_Sales'].sum(),df21['JP_Sales'].sum(),df21['Other_Sales'].sum(),df21['Global_Sales'].sum()]},index=index)



salcent
sns.set_style("white")



salcent.plot(kind='bar',figsize=(12,8),color=['#6495ED','yellowgreen'])

plt.xticks(rotation=0)

plt.title('Sales : 20th Century VS 21st Century',fontdict={'fontsize':12.5})

plt.xlabel('Country',fontdict={'fontsize':12})

plt.ylabel('Sales',fontdict={'fontsize':12});

plt.show()
fig=plt.figure(figsize=(24,8.5))

sns.set_style("white")



plt.subplot(1, 2, 1)

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('20th Century: Games with Highest Global Sales',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x='Name', y='Global_Sales', data=df20[['Name','Global_Sales']].sort_values(by='Global_Sales',ascending=False).head(20),ci=None,palette='autumn');

plt.xlabel('Game',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});



plt.subplot(1, 2, 2)

plt.title('21st Century: Games with Highest Global Sales',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xticks(rotation=90)

sns.barplot(x='Name', y='Global_Sales', data=df21[['Name','Global_Sales']].sort_values(by='Global_Sales',ascending=False).head(20),ci=None,palette='summer');

plt.xlabel('Game',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});
fig=plt.figure(figsize=(23,17))

plt.subplots_adjust(left=None, wspace=0.20, hspace=0.30)

sns.set_style("white")



plt.subplot(2, 2, 1)

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('20th Century: Number of Games released in Different Platforms',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x=df20.Platform.value_counts().index, y=df20.Platform.value_counts(),palette='autumn');

plt.xlabel('Platform',fontdict={'fontsize':16})

plt.ylabel('Number of Games',fontdict={'fontsize':16});



plt.subplot(2, 2, 2)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('21st Century: Number of Games released in Different Platforms',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(rotation=90)

sns.barplot(x=df21.Platform.value_counts().index, y=df21.Platform.value_counts(),palette='summer');

plt.xlabel('Platform',fontdict={'fontsize':16})

plt.ylabel('Number of Games',fontdict={'fontsize':16});



plt.subplot(2, 2, 3)

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('20th Century: Platforms with Highest Global Sales',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x='Platform', y='Global_Sales',data=df20.groupby('Platform').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='autumn');

plt.xlabel('Platform',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});



plt.subplot(2, 2, 4)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('21st Century: Platforms with Highest Global Sales',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(rotation=90)

sns.barplot(x='Platform', y='Global_Sales',data=df21.groupby('Platform').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='summer');

plt.xlabel('Platform',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});
fig=plt.figure(figsize=(22,16.5))

plt.subplots_adjust(left=None, wspace=0.20, hspace=0.45)

sns.set_style("white")



plt.subplot(2, 2, 1)

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('20th Century: Number of Games released in Different Genre',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x=df20.Genre.value_counts().index, y=df20.Genre.value_counts(),palette='autumn');

plt.xlabel('Genre',fontdict={'fontsize':16})

plt.ylabel('Number of Games',fontdict={'fontsize':16});



plt.subplot(2, 2, 2)

plt.title('21st Century: Number of Games released in Different Genre',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xticks(rotation=90)

sns.barplot(x=df21.Genre.value_counts().index, y=df21.Genre.value_counts(),palette='summer');

plt.xlabel('Genre',fontdict={'fontsize':16})

plt.ylabel('Number of Games',fontdict={'fontsize':16});



plt.subplot(2, 2, 3)

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('20th Century: Genre with Highest Global Sales',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x='Genre', y='Global_Sales',data=df20.groupby('Genre').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='autumn');

plt.xlabel('Genre',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});



plt.subplot(2, 2, 4)

plt.title('21st Century: Genre with Highest Global Sales',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xticks(rotation=90)

sns.barplot(x='Genre', y='Global_Sales',data=df21.groupby('Genre').sum().Global_Sales.sort_values(ascending=False).reset_index(),palette='summer');

plt.xlabel('Genre',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});
fig=plt.figure(figsize=(23,22))

plt.subplots_adjust(left=None, wspace=0.20, hspace=0.95)

sns.set_style("white")



plt.subplot(2, 2, 1)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.set_style("white")

plt.title('20th Century: Number of Games released by Different Publishers',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x=df20.Publisher.value_counts()[df20['Publisher'].value_counts()>20].index, y=df20.Publisher.value_counts()[df20['Publisher'].value_counts()>20],palette='autumn');

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('Number of Games',fontdict={'fontsize':16});



plt.subplot(2, 2, 2)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('21st Century: Number of Games released by Different Publishers',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(rotation=90)

sns.barplot(x=df21.Publisher.value_counts()[df21['Publisher'].value_counts()>150].index, y=df21.Publisher.value_counts()[df21['Publisher'].value_counts()>150],palette='summer');

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('Number of Games',fontdict={'fontsize':16});



plt.subplot(2, 2, 3)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.set_style("white")

plt.title('20th Century: Publishers with Higest Global Sales',fontdict={'fontsize':16})

plt.xticks(rotation=90)

sns.barplot(x='Publisher', y='Global_Sales',data=df20.groupby('Publisher').sum().Global_Sales.sort_values(ascending=False).reset_index()[df20.groupby('Publisher').sum().Global_Sales.sort_values(ascending=False).reset_index()['Global_Sales']>10],palette='autumn');

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});



plt.subplot(2, 2, 4)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('21st Century: Publishers with Higest Global Sales',fontdict={'fontsize':16})

sns.set_style("white")

plt.xticks(rotation=90)

sns.barplot(x='Publisher', y='Global_Sales',data=df21.groupby('Publisher').sum().Global_Sales.sort_values(ascending=False).reset_index()[df21.groupby('Publisher').sum().Global_Sales.sort_values(ascending=False).reset_index()['Global_Sales']>30],palette='summer');

plt.xlabel('Publisher',fontdict={'fontsize':16})

plt.ylabel('Global Sales',fontdict={'fontsize':16});