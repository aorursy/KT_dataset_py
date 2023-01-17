# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sn

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import iplot

import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns',None)

df = pd.read_csv('/kaggle/input/fifa19/data.csv')

df.head()
df = df.drop(['Unnamed: 0', 'ID','Photo','Flag','Club Logo','Special','Body Type', 'Real Face'],axis = 1)

df.head()
df_madrid = df.loc[(df['Club']=='Real Madrid')]

df_barca  = df.loc[(df['Club']=='FC Barcelona')]

df_juve =  df.loc[(df['Club'] == 'Juventus')]

df_psg = df.loc[(df['Club'] == 'Paris Saint-Germain')]

df_mancity = df.loc[(df['Club'] == 'Manchester City')]

df_chel = df.loc[(df['Club'] == 'Chelsea')]

df_bayern = df.loc[(df['Club'] == "FC Bayern München" )] 

df_liver = df.loc[(df['Club'] == 'Liverpool')]

df_atm = df.loc[(df['Club'] == 'Atlético Madrid')]
df_list = [df_madrid,df_barca,df_chel,df_atm,df_bayern,df_juve,df_liver,df_mancity,df_psg]

clublist = ['FC Barcelona', 'Juventus', 'Paris Saint-Germain', 'Manchester City', 'Chelsea', 'Real Madrid',

       'Atlético Madrid', 'FC Bayern München','Liverpool']
for i,j  in zip(df_list,clublist):

    print(f' club <{j}> dataframe dataframe shape is : {i.shape}')

    print("="*75)
def unique(df):

    for i in df.columns:

        print(f' feature <{i}> has {df[i].unique()} values')

        print('='*100)



        

def valuecounts(df):

    for i in df.columns:

        print(f' feature <{i}> has {df[i].value_counts()} values')

        print('='*75)

def drop(df):

    drop_list = ['LS', 'ST',

       'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM',

       'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB']

    for i in drop_list:

        df.drop(i,axis = 1,inplace = True)
for i in df_list:

    drop(i)

    
for i in df_list:

    print(f' columns are {i.columns}')

    print("="*75)
for i in df_list:

    i.drop(columns = ['RCB','RB','Loaned From'],axis = 1,inplace = True)

    
df_list = [df_barca,df_juve,df_psg,df_mancity,df_chel,df_madrid,df_atm,df_bayern,df_liver]
def barname(df,j):

    fig = px.bar(data_frame=df,x = 'Name',y = 'Overall',labels={'x':'Name of the player','y':'Overall'},

                color_discrete_sequence=[j],opacity=0.8)

    fig.show()
color_list = ['magenta','blue','purple','red','purple', 'red', 'rosybrown',

            'royalblue', 'rebeccapurple']

for i,j,k in zip(df_list,color_list,clublist):

    print(f' players of the club <{k}> with their overall ratings are shown ↓')

    barname(i,j)

    print("="*75)
df[df['Preferred Foot'] == 'Left'][['Name','Age','Club','Nationality']].head(10).style.background_gradient('magma')
df[df['Preferred Foot']=='Right'][['Name','Age','Club','Nationality']].head(10).style.background_gradient('inferno')
def barnamerep(df,j):

    fig = px.bar(data_frame=df,x = 'Name',y = 'International Reputation',labels={'x':'Name of the player','y':'International Reputation'},

                color_discrete_sequence=[j],opacity=0.8)

    fig.show()
color_list = ['magenta','blue','purple','red','purple', 'red', 'rosybrown',

            'royalblue', 'rebeccapurple']

for i,j,k in zip(df_list,color_list,clublist):

    print(f' players of the club <{k}> with their overall ratings are shown ↓')

    barnamerep(i,j)

    print("="*75)
def barsub(df):

    

    plt.figure(figsize=(16,9))

    trace1 = go.Bar(

                    x = df.Name,

                    y = df.Value,

                   name = 'Value of the player',

                   #marker = dict(color),

                   text = df.Name,

                   )

    trace2 = go.Bar(

                    x = df.Name,

                    y = df.Potential,

                    name = 'Potential of the player',

                    #marker = dict(color = 'fuchisa'),

                    xaxis = 'x2',

                    yaxis = 'y2',

                    text = df.Name,

                   )

    trace3 = go.Bar(x = df.Name,

                   y  =df.Age,

                   name =  'Age of the player',

                   xaxis = 'x3',

                   yaxis = 'y3',

                   text = df.Name,

                   )

                      

    trace4 = go.Bar(x = df.Name,

                   y = df.Wage,

                   name = 'wage of the player',

                   xaxis = 'x4',

                   yaxis = 'y4',

                   text = df.Name,

                   )

    data = [trace1,trace2,trace3,trace4];

    layout = go.Layout( 

             xaxis = dict(domain = [0,0.45]),

             xaxis2 = dict(domain = [0.55,1]),

             xaxis3 = dict(domain = [0,0.45]),

             xaxis4 = dict(domain = [0.55,1]),

             yaxis = dict(domain =  [0,0.45]),

             yaxis2 = dict(domain = [0,0.45],anchor = 'x2'),

             yaxis3 = dict(domain = [0.55,1],anchor = 'x3'),

             yaxis4 = dict(domain = [0.55,1],anchor = 'x4'),

    );

    fig = go.Figure(data=data,layout=layout)

    iplot(fig)
for i,j in zip(df_list,clublist):

    print(f' Market value and Potential of players of the <{j}> club is shown below ↓')

    barsub(i)

    print("="*75)
def workrate(df,i):

    plt.figure(figsize=(15,7))

    plt.style.use('tableau-colorblind10')

    

    sn.countplot(x = i,data=df,palette='dark')

    plt.title('Different work rates of the Players of there respective clubs', fontsize = 20)

    plt.xlabel('Work Rates of players',fontsize = 16)

    plt.ylabel('Count of the Players',fontsize = 16)

    plt.show()
for i,j in zip(df_list,clublist):

    print(f' workrate  of players of club <{j}> is shown below ↓')

    workrate(i,'Work Rate')

    print("="*75)
df_gk = df.loc[(df['Position'] == 'GK')]

df_gk.head()
df[(df['Preferred Foot'] == 'Left') & (df['Position'] == 'GK')][['Name','Age','Nationality','Club']].head(10).style.background_gradient('inferno')
df[(df['Preferred Foot'] == 'Right') & (df['Position'] == 'GK')][['Name','Age','Nationality','Club']].head(10).style.background_gradient('inferno')
df_gk = df_gk.sort_values(['Overall'],ascending=False)

df_gk[df_gk['Position']=='GK'][['Name','Club','Nationality','Age']].head(10).style.background_gradient('plasma')
df_gk = df_gk.iloc[:100,:]

df_gk.head(2)
x = df_gk.Name



trace1 = {

    'x': x,

    'y': df_gk.Value,

    'name': 'Value of the Player',

    'type': 'bar',

};

trace2 = {

    'x': x,

    'y': df_gk.Wage,

    'name': 'Wage of the player',

    'type': 'bar',

};



data = [trace1,trace2];



layout = {

          'xaxis':{'title':'Name of the player'},

          'barmode':'relative',

          'title':'Market value and Wage of the GoalKeeper'

         };



fig = go.Figure(data=data,layout=layout)

iplot(fig)

def nationality(df,k):

    fig = px.bar(data_frame=df,x = 'Name',y = 'Nationality',labels={'x':'Name','y':'Nationality'},

                color_discrete_sequence=[k],opacity=0.8)

    fig.show()
for i,j,k in zip(df_list,clublist,color_list):

    print(f' players in the club <{j}> nationality is ↓')

    nationality(i,k)

    print('='*75)
def mean_list(df):



    sum_attack = 0

    sum_defence = 0

    attack_list = ['Crossing', 'Finishing',

           'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',

           'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

           'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

           'Jumping', 'Stamina']

    defence_list = ['Strength', 'LongShots', 'Aggression',

           'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

           'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

           'GKKicking', 'GKPositioning', 'GKReflexes']

    for i,j in zip(attack_list,defence_list):

        sum_attack+=(df[i].mean())

        sum_defence+=(df[j].mean())

        df['total_attack'] = (sum_attack/len(attack_list))

        df['total_defence'] = (sum_defence/len(defence_list))

for i in df_list:

    mean_list(i)
def attackdef(df):

    trace1 = {

        'x':df.Club,

        'y':df.total_attack,

        'name': 'Total Attack',

        'type': 'bar',

    };

    trace2 = {

        'x':df.Club,

        'y':df.total_defence,

        'name': 'Total Defense',

        'type': 'bar',

    };

    data = [trace1,trace2];

    layout = {'xaxis':{'title':'Name of the club'},

             'barmode':'relative',

             'title':'Total attack and defense of the club'};

    fig = go.Figure(data=data,layout=layout)

    iplot(fig)
for i,j in zip(df_list,clublist):

    print(f' attack-defense ratio of the club <{j}> is shown below ↓')

    attackdef(i)

    print("="*75)