# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")

data.info()
data.head(20)
data.dtypes
data.columns
f,ax = plt.subplots(figsize=(25, 25))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
tm = data.groupby('Nationality').count()['ID'].sort_values(ascending = False)

plt_data = [go.Bar(

    x = tm.index,

    y = tm

    )]

layout = go.Layout(

    autosize=False,

    width=5000,

    height=600,

    title = "Total players from a Nation in the whole game"

)

fig = go.Figure(data=plt_data, layout=layout)

iplot(fig)
melted = pd.melt(frame=data,id_vars = 'Name', value_vars= ['Age','Finishing'])

melted

data1 = data['Age'].head()

data2= data['Finishing'].head()

conc_data_col = pd.concat([data1,data2],axis =1)

conc_data_col
data1 = data['Age'].tail()

data2= data['Finishing'].tail()

conc_data_col = pd.concat([data1,data2],axis =1)

conc_data_col
tm = data['Preferred Foot'].value_counts()

plt_data = [go.Bar(

    x = tm.index,

    y = tm

    )]

layout = go.Layout(

    autosize=False,

    width=500,

    height=500,

    title = "Count of players prefered foot"

)

fig = go.Figure(data=plt_data, layout=layout)

iplot(fig)
forwards = ['ST','LF','RF','CF','LW','RW']

midfielders = ['CM','LCM','RCM','RM','LM','CDM','LDM','RDM','CAM','LAM','RAM','LCM','RCM']

defenders = ['CB','RB','LB','RCB','LCB','RWB','LWB'] 

goalkeepers = ['GK']

data['Overall_position'] = None

forward_players = data[data['Position'].isin(forwards)]

midfielder_players = data[data['Position'].isin(midfielders)]

defender_players = data[data['Position'].isin(defenders)]

goalkeeper_players = data[data['Position'].isin(goalkeepers)]

data.loc[forward_players.index,'Overall_position'] = 'forward'

data.loc[defender_players.index,'Overall_position'] = 'defender'

data.loc[midfielder_players.index,'Overall_position'] = 'midfielder'

data.loc[goalkeeper_players.index,'Overall_position'] = 'goalkeeper'



tm = data['Overall_position'].value_counts()

plt_data = [go.Bar(

    x = tm.index,

    y = tm

    )]

layout = go.Layout(

    autosize=True,

    width=500,

    height=500,

    title = "Total players playing in the Overall position"

)

fig = go.Figure(data=plt_data, layout=layout)

iplot(fig)



plt.figure(figsize = (16, 12))

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)

ax = sns.countplot('Position', data = data, color = 'blue')

ax.set_xlabel(xlabel = 'Different Positions', fontsize = 16)

ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)

ax.set_title(label = 'Positions and Players', fontsize = 16)

plt.show()
plt.figure(figsize = (32, 20))

fig, axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind = "hist",y = "Penalties",bins = 50,range= (0,250),normed = True,ax = axes[0])

data.plot(kind = "hist",y = "Penalties",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt.show()
boolean = data.Potential > 93

data[boolean]
data[data["Nationality"] == "Turkey"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Club']].head(50)
data[data["Club"] == "Real Madrid"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Nationality']].head(12)
data[['Name', 'Age', 'Wage', 'Value', 'Nationality']].max()
data[['Name', 'Age', 'Wage', 'Value', 'Nationality' ]].min()
data.sort_values(by = 'Age' , ascending = True)[['Name', 'Age', 'Wage']].set_index('Name').sample(10)
data[data["Position"] == "ST"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Nationality']].head()
data.Position.unique()
sns.swarmplot(x="Dribbling", y="Finishing",hue="Preferred Foot",data = data, color = 'red')

plt.show()
sns.swarmplot(x="Position", y="Finishing",hue="Preferred Foot", data=data)

plt.show()
data.describe()
sns.countplot(x="Age", data=data)

data.loc[:,'Age'].value_counts()

plt.show()
data['Club'].fillna('No Club', inplace = True)

data['Club'].value_counts(dropna = False)
data['Preferred Foot'].fillna('Right', inplace = True)

data['Preferred Foot'].value_counts(dropna = False)