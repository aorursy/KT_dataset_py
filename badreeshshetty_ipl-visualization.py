# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (15, 5),

         'axes.labelsize': 'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}

pylab.rcParams.update(params)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import plotly

plotly.tools.set_credentials_file(username='badreeshshetty', api_key='x7GXmMWZWnhlGypX6C4z')

import plotly.graph_objs as go

import plotly.plotly as py

import cufflinks

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot

cufflinks.go_offline()



# Set global theme

cufflinks.set_config_file(world_readable=True, theme='pearl')

# Any results you write to the current directory are saved as output.
df=pd.read_excel("../input/iplallseasons_refined.xlsx")
df.head()
df.shape
df.describe()
df.info()
df.isnull().any().sum()
null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
df[df.isnull().any(axis=1)][null_columns].head()
df.loc[(df["Toss_winner"]=="no toss") & (df["Toss_decision"]=="no"),["Team1_score"]]="Abandon"

df.loc[(df["Toss_winner"]=="no toss") & (df["Toss_decision"]=="no"),["Team2_score"]]="Abandon"

df.loc[(df["Toss_winner"]=="no toss") & (df["Toss_decision"]=="no"),["Winning_margin"]]="Abandon"
df[(df["Toss_winner"]=="no toss") & (df["Toss_decision"]=="no")]
#df["Winning_margin"].isnull().sum()

df.loc[df["Winning_margin"].isnull()]
df.at[285, 'Team1_score'] = "Abandon"

df.at[285, 'Team2_score'] = "Abandon"

df.at[285, 'Winning_margin'] = "Abandon"

df.at[68, 'Winning_margin'] = "Tie"

df.at[133, 'Winning_margin'] = "Tie"

df.at[334, 'Winning_margin'] = "Tie"

df.at[348, 'Winning_margin'] = "Tie"

df.at[422, 'Winning_margin'] = "Tie"

df.at[481, 'Winning_margin'] = "Tie"

df.at[618, 'Winning_margin'] = "Tie"

df.at[492, 'Toss_winner'] = "Royal Challengers Bangalore"

df.at[492, 'Toss_decision'] = "bat"

df.at[492, 'Team2_score'] = "Did Not Play"

df.at[492, 'Winning_margin'] = "No result"

df.at[245, 'Team2_score'] = "Did Not Play"

df.at[245, 'Winning_margin'] = "No result"

df.at[518, 'Winning_margin'] = "No result"

df.at[518, 'Winning_margin'] = "No result"

df.at[488, 'Toss_winner'] = "Abandon"

df.at[488, 'Toss_decision'] = "Abandon"

df.at[488, 'Team1_score'] = "Abandon"

df.at[488, 'Team2_score'] = "Abandon"

df.at[488, 'Winning_margin'] = "Abandon"

df.isnull().any().sum()
df
df["Match_venue"] = df["Match_venue"].replace(['M Chinnaswamy Stadium, Bangalore'],\

                                              'M Chinnaswamy Stadium, Bengaluru')

df["Match_venue"] = df["Match_venue"].replace(['Punjab Cricket Association Stadium, Mohali, Chandigarh'], \

                                              'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh')

df["Match_venue"] = df["Match_venue"].replace(['Sharjah Cricket Stadium'], \

                                               'Sharjah Cricket Stadium, Dubai')

df["Match_venue"] = df["Match_venue"].replace(['Dubai International Cricket Stadium'], \

                                               'Dubai International Cricket Stadium, Dubai')

df["Team1"] = df["Team1"].replace(['Rising Pune Supergiants'], \

                                               'Rising Pune Supergiant')

df["Team2"] = df["Team2"].replace(['Rising Pune Supergiants'], \

                                               'Rising Pune Supergiant')

df["Winning_team"] = df["Winning_team"].replace(['Rising Pune Supergiants'], \

                                               'Rising Pune Supergiant')



ind_city=[' Bengaluru', ' Mohali',' Delhi',' Kolkata',' Mumbai',' Jaipur', ' Uppal',' Chepauk',\

        ' Motera',' Cuttack',' Jamtha',' Dharamsala',' Kochi',' Indore',' Visakhapatnam',' Pune',' Raipur',\

         ' Ranchi']

sa_city=[' Cape Town',' Port Elizabeth',' Durban',' Centurion',' East London',' Johannesburg',\

        ' Kimberley'' Bloemfontein']



#def country(venue):

   # if ind_city == venue:

 #       return "India"

 #   elif sa_city == venue:

#        return "South Africa"

#    elif "Abi Dhabi" == venue:

#        return "Abu Dhabi"

 #  else:

   #     return "Dubai"

    

#df["Country"] = df["Match_venue"].apply(country)



df.loc[df['Match_venue'].str.contains('|'.join(ind_city)), 'Country'] = 'India'

df.loc[df['Match_venue'].str.contains('|'.join(sa_city)), 'Country'] = 'South Africa'

df.loc[df['Match_venue'].str.contains('Dubai'), 'Country'] = 'Dubai'

df.loc[df['Match_venue'].str.contains('Abu Dhabi'), 'Country'] = 'Abu Dhabi'

df
df['Country'].unique()
df['Match_venue'].str.split(",").str[1].unique().tolist()
df["Stadium"]=df["Match_venue"].str.split(",").str[0]

df["City"]=df["Match_venue"].str.split(",").str[1]
df.head()
df["Match_time"].unique()
df.loc[df["Match_time"]==" ","Match_time"]="day/night"
df["Match_time"].unique()
df["Toss_winner"].unique()
df.loc[df["Toss_winner"]=="no toss","Toss_winner"]="No Toss"

df.loc[df["Toss_decision"]=="no","Toss_decision"]="No Toss"
df["Toss_decision"].unique()
df.loc[df["Winning_team"]=="Match abandoned without a ball bowled"]
df.loc[df["Toss_decision"]=="Rajasthan","Toss_decision"]="field"


df["Toss_decision"].unique()
df["Team1"].unique()
df["Team2"].unique()
df["Winning_team"].unique()
df.loc[df["Winning_team"].str.contains("Match tied")]
df.at[68, 'Winning_team'] = "Tie (Rajasthan Royals,Kolkata Knight Riders)"

df.at[133, 'Winning_team'] = "Tie (Kings XI Punjab,Chennai Super Kings)"

df.at[334, 'Winning_team'] = "Tie (Sunrisers Hyderabad,RCB)"

df.at[348, 'Winning_team'] = "Tie (RCB,Delhi Daredevils)"

df.at[422, 'Winning_team'] = "Tie (Rajasthan Royals, Kolkata Knight Riders)"

df.at[481, 'Winning_team'] = "Tie (Kings XI Punjab, Rajasthan Royals)"

df.at[618, 'Winning_team'] = "Tie (Mumbai Indians, Gujarat Lions)"

df.loc[df["Winning_team"].str.contains("Tie")]
df["Winning_margin"].unique()
df
df.loc[df['Match_date'].str.contains('2008'), 'Year'] = '2008'

df.loc[df['Match_date'].str.contains('2009'), 'Year'] = '2009'

df.loc[df['Match_date'].str.contains('2010'), 'Year'] = '2010'

df.loc[df['Match_date'].str.contains('2011'), 'Year'] = '2011'

df.loc[df['Match_date'].str.contains('2012'), 'Year'] = '2012'

df.loc[df['Match_date'].str.contains('2013'), 'Year'] = '2013'

df.loc[df['Match_date'].str.contains('2014'), 'Year'] = '2014'

df.loc[df['Match_date'].str.contains('2015'), 'Year'] = '2015'

df.loc[df['Match_date'].str.contains('2016'), 'Year'] = '2016'

df.loc[df['Match_date'].str.contains('2017'), 'Year'] = '2017'

df.loc[df['Match_date'].str.contains('2018'), 'Year'] = '2018'
df['Match_date'].unique()
df.loc[df['Match_date'].str.contains('Apr'), 'Month'] = 'April'

df.loc[df['Match_date'].str.contains('May'), 'Month'] = 'May'

df.loc[df['Match_date'].str.contains('Mar'), 'Month'] = 'March'

df.loc[df['Match_date'].str.contains('Jun'), 'Month'] = 'June'

#df["Toss&MatchWinner"]=[1 if df['Toss_winner'] == df['Winning_team'] else 0]

df["Toss&MatchWinner"] = np.where(df['Toss_winner'] == df['Winning_team'], "Won both Toss & Match", "Won Toss but Lost Match")
df.head()
df["Match_number"].unique()
df
fin_win=df.loc[df["Match_number"]=="Final"]["Winning_team"].value_counts()
#plt.figure(figsize=(12,10))

#plt.bar(fin_win.index,fin_win.values)

#plt.title("IPL Final Winners")

#plt.xlabel("Teams")

#plt.ylabel("No. of Times won Final")

#plt.xticks(rotation=25);

data = [go.Bar(

            x=fin_win.index,

            y=fin_win.values

    )]



iplot(data, filename='final_win-bar')

df.head()
month_play=df["Month"].value_counts()
#plt.figure(figsize=(12,10))

#explode = (0, 0, 0.1, 0)  

#plt.pie(x=month_play.values,labels=month_play.index,\

   #     autopct='%.2f',explode=explode)

#plt.axis('equal');

labels = month_play.index

values = month_play.values



trace = go.Pie(labels=labels, values=values)



iplot([trace], filename='month_play_pie_chart')
match_time_play=df["Match_time"].value_counts()
labels = match_time_play.index

values = match_time_play.values

trace = go.Pie(labels=labels, values=values)

iplot([trace], filename='match_time_play_pie_chart')
#df.loc[df["Match_time"]=="night","Toss_decision"].value_counts()

dnb_winners=df.loc[(df["Match_time"]=="day/night")&(df["Toss_decision"]=="bat"),"Winning_team"].value_counts()

dnf_winners=df.loc[(df["Match_time"]=="day/night")&(df["Toss_decision"]=="field"),"Winning_team"].value_counts()

db_winners=df.loc[(df["Match_time"]=="night")&(df["Toss_decision"]=="bat"),"Winning_team"].value_counts()

df_winners=df.loc[(df["Match_time"]=="night")&(df["Toss_decision"]=="field"),"Winning_team"].value_counts()
trace0 = go.Bar(

    x=db_winners.index,

    y=db_winners.values,

    name='Day Match and Decided to Bat',

    marker=dict(

        color='rgb(49,130,189)'

    )

)

trace1 = go.Bar(

    x=df_winners.index,

    y=df_winners.values,

    name='Day Match and Decided to Field',

    marker=dict(

        color='rgb(204,204,204)',

    )

)



data = [trace0, trace1]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='group',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='Day Match-bar')
trace0 = go.Bar(

    x=dnb_winners.index,

    y=dnb_winners.values,

    name='Day/Night Match and Decided to Bat',

    marker=dict(

        color='rgb(49,130,189)'

    )

)

trace1 = go.Bar(

    x=dnf_winners.index,

    y=dnf_winners.values,

    name='Day/Night Match and Decided to Field',

    marker=dict(

        color='rgb(204,204,204)',

    )

)



data = [trace0, trace1]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='group',

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='Day/Night Match-bar')
stad_play=df["Stadium"].value_counts()
dnb_winners=df.loc[(df["Match_time"]=="day/night")&(df["Toss_decision"]=="bat"),"Winning_team"].value_counts()
data = [go.Bar(

            x=stad_play.index,

            y=stad_play.values

    )]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename='stad_play-bar')
city_play=df["City"].value_counts()
data = [go.Bar(

            x=city_play.index,

            y=city_play.values

    )]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig,filename='stad_play-bar')
#["Greater than 150" if (df['Team1_score'].str.split("/").str[0]).astype(int) > 150 else "Less than 150"]
df['Team1_score'].str.split("/").str[1].tolist()
df['Team2_score'].str.split("/").str[0].tolist()
df['Team2_score'].str.split("/").str[1].tolist()