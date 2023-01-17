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
from plotly.offline import iplot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.max_columns',None)
df = pd.read_csv('/kaggle/input/top-women-chess-players/top_women_chess_players_aug_2020.csv')
df.head()
data = df.loc[:100,:]
data.head()
data = data.drop(['Fide id','Gender','Inactive_flag'],axis = 1)
data.head(3)
def replacement(value):
    out = value.replace(',','')
    return out
data['Name'] = data['Name'].apply(lambda x: replacement(x))
data.head(2)
px.bar(data_frame=data,x = 'Name',y = 'Standard_Rating',labels={'x':'Name','y':'Standard Rating'},
      color_discrete_sequence=['purple'],opacity=1)
data['Curr_year'] = 2020
data['Age'] = data['Curr_year']-data['Year_of_birth']
data.head(2)
def barsubplots(df):
    
    trace1 = go.Bar(x = df.Name,
                   y = df.Standard_Rating,
                   name = 'Standard Rating of the Player',
                   text = df.Name,
                   )
    trace2 = go.Bar(x = df.Name,
                    y = df.Rapid_rating,
                   name = 'Rapid Rating of the Player',
                   text = df.Name,
                   xaxis = 'x2',
                   yaxis = 'y2',
                   )
    trace3 = go.Bar(x = df.Name,
                   y = df.Blitz_rating,
                   name = 'Blitz Rating of the Player',
                   text = df.Name,
                   xaxis = 'x3',
                   yaxis = 'y3',
                   )
    trace4 = go.Bar(x = df.Name,
                    y = df.Age,
                   name = 'Age of the Player',
                   text = df.Name,
                   xaxis = 'x4',
                   yaxis = 'y4',
                   )
    data = [trace1,trace2,trace3,trace4];
    layout = go.Layout(xaxis=dict(domain = [0,0.45]),
                       xaxis2 = dict(domain = [0.55,1]),
                       xaxis3 = dict(domain = [0,0.45]),
                       xaxis4 = dict(domain = [0.55,1]),
                       yaxis = dict(domain = [0,0.45]),
                       yaxis2 = dict(domain = [0,0.45],anchor = 'x2'),
                       yaxis3 = dict(domain = [0.55,1],anchor = 'x3'),
                       yaxis4 = dict(domain = [0.55,1],anchor = 'x4'),
                      )
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)
barsubplots(data)
sn.countplot(data=data,x= 'Title',palette='plasma')

df_gm = data.loc[(data['Title'] == 'GM')]
df_gm.shape
barsubplots(df_gm)
sn.set_palette('plasma')
plt.figure(figsize=(16,9))
sn.set_style(style='darkgrid')
sn.countplot(data=data,x = 'Federation',palette='plasma')
sn.set_palette('plasma')
plt.figure(figsize=(16,9))
sn.set_style(style='darkgrid')
sn.countplot(data=df_gm,x = 'Federation',palette='plasma')
df_gm[df_gm['Federation'] == 'IND'].style.background_gradient('plasma')
df_gm[df_gm['Federation'] == 'CHN'].style.background_gradient('plasma')
df_gm[df_gm['Federation'] == 'RUS'].style.background_gradient('plasma')
df_gm[(df_gm['Age']>=20) & (df_gm['Age']<=23)].style.background_gradient('plasma')
def charts(df,i):
    chart = px.pie(df,values = i,names = 'Name',height = 600)
    chart.update_traces(textposition = 'inside',textinfo = 'percent+label')
    
    chart.update_layout(title_x = 0.5,
                       geo = dict(
                       showframe = True,
                       showcoastlines = False,))
    chart.show()
rating = ['Standard_Rating','Rapid_rating','Blitz_rating']
for i in rating:
    print(f' plots for the <{i}> are shown below ↓')
    charts(df_gm,i)
    print("="*75)
