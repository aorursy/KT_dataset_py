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
# Data Manipulation 

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import seaborn as sns



#datetime

import datetime as dt



#warnings

import warnings

warnings.filterwarnings("ignore")





#plotly

import plotly.graph_objects as go

import plotly.figure_factory as ff

import plotly.express as px

from plotly.subplots import make_subplots
fifa=pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

fifa.head()
fifa.dob=pd.to_datetime(fifa.dob)
fifa_potential=fifa[(fifa.potential>85 )& (fifa.overall<80)]

fifa_potential.head()
fifa_potential_ready=fifa_potential[(fifa_potential.overall<80)&(fifa_potential.overall>70)]

fifa_potential.head()
position="GK"

fifa_potential_ready_GK=fifa_potential_ready[fifa.player_positions.str.contains(position)]

fifa_potential_ready_GK
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.overall,text=fifa_potential_ready_GK.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.potential,text=fifa_potential_ready_GK.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential GK in FIFA 20',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show() 
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_ready_GK.short_name, y=fifa_potential_ready_GK.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential GK player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
position="LB"

position1="LWB"

fifa_potential_LB=fifa_potential_ready[(fifa.player_positions.str.contains(position))|(fifa.player_positions.str.contains(position1))]

fifa_potential_LB
fig = go.Figure(data=[

            go.Bar(name='overall', x=fifa_potential_LB.short_name, y=fifa_potential_LB.overall,text=fifa_potential_LB.overall,textposition='auto'),

            go.Bar(name='potential', x=fifa_potential_LB.short_name, y=fifa_potential_LB.potential,text=fifa_potential_LB.potential,textposition='auto')

            ])

fig.update_layout(title='Top potential LB in FIFA 20 ',

                   xaxis_title='player name ',

                   yaxis_title='Rating')

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=fifa_potential_LB.short_name, y=fifa_potential_LB.value_eur,

                    mode='lines+markers',

                    ))

fig.update_layout(title=' Top Potential LB player values(Euros)',

                   xaxis_title='player name ',

                   yaxis_title='Value')

fig.show()
import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import skew



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
data.shape
numerical_features   = data.select_dtypes(include = [np.number]).columns

categorical_features = data.select_dtypes(include= [np.object]).columns
numerical_features
categorical_features
data.describe()
data.hist(figsize=(30,30))

plt.plot()
skew_values = skew(data[numerical_features], nan_policy = 'omit')

pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 

           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)
# Missing values

missing_values = data.isnull().sum().sort_values(ascending = False)

percentage_missing_values = (missing_values/len(data))*100

pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
# encode target variable 'Man of the match' into binary format

data['Man of the Match'] = data['Man of the Match'].map({'Yes': 1, 'No': 0})
sns.countplot(x = 'Man of the Match', data = data)
plt.figure(figsize=(30,10))

sns.heatmap(data[numerical_features].corr(), square=True, annot=True,robust=True, yticklabels=1)
var = ['Man of the Match','Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 

       'Fouls Committed', 'Own goal Time']

corr = data.corr()

corr = corr.filter(items = ['Man of the Match'])

plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True)
var = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 

       'Fouls Committed', 'Own goal Time']

plt.figure(figsize=(15,10))

sns.heatmap((data[var].corr()), annot=True)
var1 = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed']

var1.append('Man of the Match')

sns.pairplot(data[var1], hue = 'Man of the Match', palette="husl")

plt.show()
dummy_data = data[var1]

plt.figure(figsize=(20,10))

sns.boxplot(data = dummy_data)

plt.show()
data.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)
categorical_features
def uniqueCategories(x):

    columns = list(x.columns).copy()

    for col in columns:

        print('Feature {} has {} unique values: {}'.format(col, len(x[col].unique()), x[col].unique()))

        print('\n')

uniqueCategories(data[categorical_features].drop('Date', axis = 1))
data.drop('Date', axis = 1, inplace=True)
data.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)
print(data.shape)

data.head()
cleaned_data  = pd.get_dummies(data)
print(cleaned_data.shape)

cleaned_data.head()