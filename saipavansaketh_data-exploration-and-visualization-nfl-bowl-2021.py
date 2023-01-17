# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt
players = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/players.csv")

players.head()
fig = px.histogram(players, x="weight",width=600,height=500,nbins=50,

                   title='Players weight distribution')

fig.show()
players.height = players.height.str.replace('-','*12+').apply(lambda x: eval(x))

df = players.groupby('height')['nflId'].count().reset_index(name = 'counts')

df = df.sort_values(by = 'counts', ascending = False)

fig = px.bar(df, x='height', y='counts', color = 'height',width=600,height=500,

             title='Players height distribution')

fig.show()
df = players.groupby('position')['nflId'].count().reset_index(name = 'counts')

df = df.sort_values(by  = 'counts', ascending = False)

fig = px.bar(df, y='position', x='counts', color = 'position',width=600,height=500,

            title=' Top Position by number of players')

fig.show()
players.birthDate = pd.to_datetime(players.birthDate)

players['birthyear'] = players['birthDate'].dt.year

df = players.groupby('birthyear')['nflId'].count().reset_index(name = 'counts')

fig = px.bar(df, x='birthyear', y='counts', color = 'birthyear',width=600,height=500

            ,title='Players Birthyear distribution')

fig.show()
games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/games.csv')

games.head()
df= games['gameDate'].value_counts().reset_index()

df.columns = ['Date' , "Games"]

df= df.sort_values('Games' , ascending = True)



fig= px.bar(df,y = 'Date',x = 'Games',orientation = 'h',color = 'Games',height=500,width=600

             ,title = 'Number of games for every Date')

fig.show()



fig= px.line(df, x='Date',y="Games",title='Line Plot',height=500,width=600)

fig.show()
df= games['gameTimeEastern'].value_counts().reset_index()

df.columns = ['Time' , 'Games']

df= df.sort_values('Games')



fig = px.bar(df, x = 'Games',y = 'Time',color = 'Games',orientation = 'h',height = 500,width = 600

            ,title = 'Number of games for every Time')

fig.show()



fig = px.line(df, x='Time',y="Games",  title='line plot',height = 500,width = 600)

fig.show()
df= games['homeTeamAbbr'].value_counts().reset_index()

df.columns = ['Team', 'Games']

df= df.sort_values('Games')



fig = px.bar(df, y='Team', x="Games", orientation='h',color = 'Games',

             title='Number of games for every team (home)', height=500, width=600)

fig.show()
df= games['visitorTeamAbbr'].value_counts().reset_index()

df.columns = ['Team', 'Games']

df = df.sort_values('Games')



fig = px.bar(df, y='Team', x="Games", orientation='h',color='Team', 

             title='Number of games for every team (Visitor)', height=500, width=600)

fig.show()
df= games['week'].value_counts().reset_index()

df.columns = ['Week_Numeric', 'Games']

df = df.sort_values('Games')



fig = px.bar(df, y='Week_Numeric', x="Games",orientation='h',color = 'Games',

             title='Number of games for every week', height=500, width=600)

fig.show()
def uni(df,col,v,hue =None):

    sns.set(style="darkgrid")

    

    if v == 0:

        fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(20,8))

        ax[0].set_title("Distribution Plot")

        sns.distplot(df[col],ax=ax[0], color="#da0463")

        plt.yscale('log')

        ax[1].set_title("Violin Plot")

        sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile", color="#f85959")

        plt.yscale('log')

        ax[2].set_title("Box Plot")

        sns.boxplot(data =df, x=col,ax=ax[2],orient='v', color="#d89cf6")

        plt.yscale('log')

        

    if v == 1:

        temp = pd.Series(data = hue)

        fig, ax = plt.subplots()

        width = len(df[col].unique()) + 6 + 4*len(temp.unique())

        fig.set_size_inches(width , 7)

        ax = sns.countplot(data = df, x= col, color="#4CB391", order=df[col].value_counts().index,hue = hue) 

        

        if len(temp.unique()) > 0:

            for p in ax.patches:

                ax.annotate('{:1.1f}%'.format((p.get_height()*100)/float(len(loan))), (p.get_x()+0.05, p.get_height()+20))  

        else:

            for p in ax.patches:

                ax.annotate(p.get_height(), (p.get_x()+0.32, p.get_height()+20)) 

        del temp

    else:

        exit     

    plt.show()
players["WeightKg"] = players["weight"]*0.45359237

uni(df=players,col='WeightKg',v=0)

uni(df=players,col='height',v=0)
df= players.collegeName.value_counts().reset_index()

df.columns = ['collegeName' , 'Players']

df.sort_values('Players' , inplace=True)



fig = px.bar(df.tail(20),y= 'Players',x= 'collegeName',

             title = 'Top 20 colleges by number of Players',color = 'Players',width=600,height=500)

fig.show()
plays = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/plays.csv")

plays.head()
df = plays.groupby("isDefensivePI")['playId'].count().reset_index(name = 'counts')

fig = px.pie(df, values='counts', names='isDefensivePI', 

             title='Defensive Count',height=600,width=600)

fig.show()
df= plays['offenseFormation'].value_counts().reset_index()

df.columns = ['offenseFormation', 'plays']

df= df.sort_values('plays')



fig = px.pie(df, names='offenseFormation', values="plays",

             title='Plays offense formation type',height=600,width=600)

fig.show()
plays = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/plays.csv")

im = plays["passResult"].value_counts()

df = pd.DataFrame({'labels': ['Complete Pass', 'Incomplete pass', 'Quarterback sack', 'Intercepted pass', 'R'],'values': im.values})

fig=px.pie(df,labels='labels',values='values', title='Pass Result Distribution', hole = 0.5,height=600,width=600)

fig.show()
im = plays["playType"].value_counts()

df = pd.DataFrame({'labels': ['Pass', 'Sack', 'Unknown'],'values': im.values})

fig=px.pie(df,labels='labels',values='values', 

           title='Play Type Distribution', hole = 0.5,height=600,width=600)

fig.show()
df= plays['possessionTeam'].value_counts().reset_index()

df.columns = ['team', 'plays']

df = df.sort_values('plays')



fig = px.bar(df, y='team', x="plays", orientation='h', color = 'plays',

             title='Number of plays for every team',height=800,width=600)

fig.show()