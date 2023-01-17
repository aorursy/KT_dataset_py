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

import seaborn as sn

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import iplot

import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns',None)

df = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv')

df.head()
df['year'] = pd.to_datetime(df['date']).dt.year

df.head(2)
data = df.loc[(df['tournament'] == 'FIFA World Cup')]

print(f' original dataframes shape is {df.shape}\n\n modified dataframes shape is {data.shape}')
df_arg = data.loc[(data['home_team'] == 'Argentina') ]

df_brazil = data.loc[(data['home_team'] == 'Brazil') ]

df_england = data.loc[(data['home_team'] == 'England') ]

df_germay = data.loc[(data['home_team'] == 'Germany')]

df_india = data.loc[(data['home_team'] == 'India') ]

df_italy = data.loc[(data['home_team'] == 'Italy') ]

df_spain = data.loc[(data['home_team'] == 'Spain') ]
df_arg[df_arg['home_team'] == 'Argentina'].style.background_gradient('plasma')
df_italy[df_italy['home_team'] == 'Italy'].style.background_gradient('plasma')
from plotly.subplots import make_subplots
def barsubplots(df):

    

    trace1 = go.Bar(x = df.away_team,

                   y = df.home_score,

                   name = 'Score against the away team',

                   text = df.away_team,

                   )

    trace2 = go.Bar(x = df.home_team,

                   y = df['away_team'].value_counts(),

                   name = 'count of away teams',

                   text = df.away_team,

                   xaxis = 'x2',

                   yaxis = 'y2',

                   )

    data = [trace1,trace2];

    layout = go.Layout(xaxis=dict(domain = [0,0.45]),

                      xaxis2 = dict(domain = [0.55,1]),

                      yaxis = dict(domain = [0,0.45]),

                      yaxis2 = dict(domain = [0,0.45],anchor = 'x2')

                      )

    fig = go.Figure(data=data,layout=layout)

    iplot(fig)
df_list = [df_arg,df_brazil,df_england,df_germay,df_italy,df_spain]

country_list = ['Argentina','Brazil','England','Germany','Italy','Spain']

for i,j in zip(df_list,country_list):

    print(f' plots for the team <{j}> in the FIFA tournament is shown below ↓')

    barsubplots(i)

    print("="*75)
def barplot(df,i,j):

    fig = px.bar(data_frame=df,x = i,y = j,labels={'x':'Team name','y':'Score'},

                color_discrete_sequence=['purple'],opacity=1)

    fig.show()
for i,j in zip(df_list,country_list):

    print(f' plots for the team <{j}> in the FIFA tournament is shown below ↓')

    barplot(i,'away_team','away_score')

    print("="*75)
df_germay[df_germay['home_team'] == 'Germany'].style.background_gradient('plasma')
year_list = data['year'].unique()
data_1991 = data.loc[(data['year'] == 1991)]

data_1995 = data.loc[(data['year'] == 1995)]

data_1999 = data.loc[(data['year'] == 1999)]

data_2003 = data.loc[(data['year'] == 2003)]

data_2007 = data.loc[(data['year'] == 2007)]

data_2011 = data.loc[(data['year'] == 2011)]

data_2015 = data.loc[(data['year'] == 2015)]

data_2019 = data.loc[(data['year'] == 2019)]
def countplot(df,i):

    plt.figure(figsize=(25,12.5))

    sn.set_style(style='darkgrid')

    sn.set_palette(palette='plasma')

    sn.countplot(data=df,x = df[i])

    plt.title('Count of the teams participated in FIFA')

    plt.xlabel('Team name')

    plt.ylabel('Home apperances')

    plt.show()
datayear = [data_1991,data_1995,data_1999,data_2003,data_2007,data_2011,data_2015,data_2019]

for i,j in zip(datayear,year_list):

    print(f' count of teams participated for the FIFA{j} is shown below ↓')

    countplot(i,'home_team')

    print("="*75)
#creating a function 

def namelist(df):

    name_list = []

    count_list = []

    value = df['home_team'].value_counts()

    for i,j in zip(value.index,value.values):

        name_list.append(i)

        count_list.append(j)

    return name_list,count_list
name_1991,count_1991 = namelist(data_1991)

name_1995,count_1995 = namelist(data_1995)

name_1999,count_1999 = namelist(data_1999)

name_2003,count_2003 = namelist(data_2003)

name_2007,count_2007 = namelist(data_2007)

name_2011,count_2011 = namelist(data_2011)

name_2015,count_2015 = namelist(data_2015)

name_2019,count_2019 = namelist(data_2019)
def zips(list1,list2):

    name_count = dict(list(zip(list1,list2)))

    return name_count
dummy1 = zips(name_1991,count_1991)

dummy2 = zips(name_1995,count_1995)

dummy3 = zips(name_1999,count_1999)

dummy4 = zips(name_2003,count_2003)

dummy5 = zips(name_2007,count_2007)

dummy6 = zips(name_2011,count_2011)

dummy7 = zips(name_2015,count_2015)

dummy8 = zips(name_2019,count_2019)
def namelistaway(df):

    name_list = []

    count_list = []

    value = df['away_team'].value_counts()

    for i,j in zip(value.index,value.values):

        name_list.append(i)

        count_list.append(j)

    return name_list,count_list
nameaway_1991,countaway_1991 = namelistaway(data_1991)

nameaway_1995,countaway_1995 = namelistaway(data_1995)

nameaway_1999,countaway_1999 = namelistaway(data_1999)

nameaway_2003,countaway_2003 = namelistaway(data_2003)

nameaway_2007,countaway_2007 = namelistaway(data_2007)

nameaway_2011,countaway_2011 = namelistaway(data_2011)

nameaway_2015,countaway_2015 = namelistaway(data_2015)

nameaway_2019,countaway_2019 = namelistaway(data_2019)
dummyaway1 = zips(nameaway_1991,countaway_1991)

dummyaway2 = zips(nameaway_1995,countaway_1995)

dummyaway3 = zips(nameaway_1999,countaway_1999)

dummyaway4 = zips(nameaway_2003,countaway_2003)

dummyaway5 = zips(nameaway_2007,countaway_2007)

dummyaway6 = zips(nameaway_2011,countaway_2011)

dummyaway7 = zips(nameaway_2015,countaway_2015)

dummyaway8 = zips(nameaway_2019,countaway_2019)
from collections import Counter

fifa1991 =  Counter(dummy1)+Counter(dummyaway1)

fifa1995 =  Counter(dummy2)+Counter(dummyaway2)

fifa1999 =  Counter(dummy3)+Counter(dummyaway3)

fifa2003 =  Counter(dummy4)+Counter(dummyaway4)

fifa2007 =  Counter(dummy5)+Counter(dummyaway5)

fifa2011 =  Counter(dummy6)+Counter(dummyaway6)

fifa2015 =  Counter(dummy7)+Counter(dummyaway7)

fifa2019 =  Counter(dummy8)+Counter(dummyaway8)
def dataframe(df):

    frame = pd.DataFrame(list(df.items()),columns=['Team','Matches_played'])

    return frame
matches_1991 = dataframe(fifa1991)

matches_1995 = dataframe(fifa1995)

matches_1999 = dataframe(fifa1999)

matches_2003 = dataframe(fifa2003)

matches_2007 = dataframe(fifa2007)

matches_2011 = dataframe(fifa2011)

matches_2015 = dataframe(fifa2015)

matches_2019 = dataframe(fifa2019)
def matchplot(df):

    fig = px.bar(data_frame=df,x = 'Team',y = 'Matches_played',color='Team',labels={'x':'Team_name','y':'Matches_played'},opacity=1)

    fig.show()
match_list = [matches_1991,matches_1995,matches_1999,matches_2003,matches_2007,matches_2011,matches_2015,matches_2019]

for i,j in zip(match_list,year_list):

    print(f' matches played stats for the FIFA{j} are shown below↓')

    matchplot(i)

    print("="*75)