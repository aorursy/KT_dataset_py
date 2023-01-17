# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = pd.read_csv('../input/fifa19/data.csv')
data.head()
data.shape
data.info()
def players_of_country(x):
    return data[data['Nationality'] == x][['Name','Overall','Potential','Position']]
players_of_country('India')
def club(x):
    return data[data['Club'] == x][['Name','Jersey Number','Position','Overall','Nationality','Age','Wage',
                                    'Value','Contract Valid Until']]
x = club('Manchester United')
x.shape
data.isnull().sum()
data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)
data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)
data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)
data['Curve'].fillna(data['Curve'].mean(), inplace = True)
data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)
data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)
data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)
data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)
data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)
data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)
data['Weight'].fillna('200lbs', inplace = True)
data['Contract Valid Until'].fillna(2019, inplace = True)
data['Height'].fillna("5'11", inplace = True)
data['Loaned From'].fillna('None', inplace = True)
data['Joined'].fillna('Jul 1, 2018', inplace = True)
data['Jersey Number'].fillna(8, inplace = True)
data['Body Type'].fillna('Normal', inplace = True)
data['Position'].fillna('ST', inplace = True)
data['Club'].fillna('No Club', inplace = True)
data['Work Rate'].fillna('Medium/ Medium', inplace = True)
data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)
data['Weak Foot'].fillna(3, inplace = True)
data['Preferred Foot'].fillna('Right', inplace = True)
data['International Reputation'].fillna(1, inplace = True)
data['Wage'].fillna('€200K', inplace = True)
data.fillna(0, inplace = True)
data.hist(figsize=(20, 20))
plt.show()
import plotly.express as px
fig = px.scatter(data, x="Acceleration",
                 color="Age",
                 size='Age', 
                 hover_data=['Age'], 
                 title = "Acceleration")
fig.show()
import plotly.graph_objects as go
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=data['Age'],
    name='sin',
    mode='markers',
    marker_color='rgba(152, 0, 0, .8)'
))
fig1.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig1.update_layout(title='Age',
                  yaxis_zeroline=False, xaxis_zeroline=False)


fig1.show()
fig2 = go.Figure(data=go.Scatter(x=data['Aggression'],
                                mode='markers',
                                marker_color=data['Age'],
                                text=data['Potential'])) # hover text goes here

fig2.update_layout(title='Aggression')
fig2.show()
