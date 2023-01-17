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

data=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

data.info()
data.head()
data['Genre'].unique()
import plotly.express as px
pie_data=data['Genre'].value_counts()
pie_data = pd.DataFrame(pie_data)
pie_data=pie_data.reset_index()

fig = px.pie(pie_data, values='Genre', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
data_sport=data[data['Genre'] == 'Sports']
data_sport=data_sport['Publisher'].value_counts()[:20]
data_sport = pd.DataFrame(data_sport)
data_sport=data_sport.reset_index()
fig = px.pie(data_sport, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

data_sport=data[data['Genre'] == 'Sports']

data_sport_NA_Sales=data_sport
data_sport_NA_Sales = pd.DataFrame(data_sport_NA_Sales)
fig = px.histogram(data_sport_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_sport_EU_Sales=data_sport
data_sport_EU_Sales = pd.DataFrame(data_sport_EU_Sales)
fig = px.histogram(data_sport_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_sport_JP_Sales=data_sport
data_sport_JP_Sales = pd.DataFrame(data_sport_JP_Sales)
fig = px.histogram(data_sport_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()


data_platform=data[data['Genre']=='Platform']
data_platform=data_platform['Publisher'].value_counts()[:20]
data_platform = pd.DataFrame(data_platform)
data_platform=data_platform.reset_index()
fig = px.pie(data_platform, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
data_platform=data[data['Genre']=='Platform']

data_platform_NA_Sales=data_platform
data_platform_NA_Sales = pd.DataFrame(data_platform_NA_Sales)
fig = px.histogram(data_platform_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_platform_EU_Sales=data_platform
data_platform_EU_Sales = pd.DataFrame(data_platform_EU_Sales)
fig = px.histogram(data_platform_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_sport_JP_Sales=data_platform
data_sport_JP_Sales = pd.DataFrame(data_sport_JP_Sales)
fig = px.histogram(data_sport_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Racing=data[data['Genre']=='Racing']
data_Racing=data_Racing['Publisher'].value_counts()[:20]
data_Racing = pd.DataFrame(data_Racing)
data_Racing=data_Racing.reset_index()
fig = px.pie(data_Racing, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
data_Racing=data[data['Genre']=='Racing']

data_NA_Sales=data_Racing
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Racing
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Racing
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Role_Playing=data[data['Genre']=='Role-Playing']
data_Role_Playing=data_Role_Playing['Publisher'].value_counts()[:20]
data_Role_Playing = pd.DataFrame(data_Role_Playing)
data_Role_Playing=data_Role_Playing.reset_index()
fig = px.pie(data_Role_Playing, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
data_Role_Playing=data[data['Genre']=='Role-Playing']

data__NA_Sales=data_Role_Playing
data__NA_Sales = pd.DataFrame(data__NA_Sales)
fig = px.histogram(data__NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data__EU_Sales=data_Role_Playing
data__EU_Sales = pd.DataFrame(data__EU_Sales)
fig = px.histogram(data__EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data__JP_Sales=data_Role_Playing
data__JP_Sales = pd.DataFrame(data__JP_Sales)
fig = px.histogram(data__JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Puzzle=data[data['Genre']=='Puzzle']
data_Puzzle=data_Puzzle['Publisher'].value_counts()[:20]
data_Puzzle = pd.DataFrame(data_Puzzle)
data_Puzzle=data_Puzzle.reset_index()
fig = px.pie(data_Puzzle, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

data_Puzzle=data[data['Genre']=='Puzzle']

data_NA_Sales=data_Puzzle
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Puzzle
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Puzzle
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Misc=data[data['Genre']=='Misc']
data_Misc=data_Misc['Publisher'].value_counts()[:20]
data_Misc = pd.DataFrame(data_Misc)
data_Misc=data_Misc.reset_index()
fig = px.pie(data_Misc, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()


data_Misc=data[data['Genre']=='Misc']

data_NA_Sales=data_Misc
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Misc
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Misc
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Shooter=data[data['Genre']=='Shooter']
data_Shooter=data_Shooter['Publisher'].value_counts()[:20]
data_Shooter = pd.DataFrame(data_Shooter)
data_Shooter=data_Shooter.reset_index()
fig = px.pie(data_Shooter, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
data_Shooter=data[data['Genre']=='Shooter']

data_NA_Sales=data_Shooter
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Shooter
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Shooter
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Simulation=data[data['Genre']=='Simulation']
data_Simulation=data_Simulation['Publisher'].value_counts()[:20]
data_Simulation = pd.DataFrame(data_Simulation)
data_Simulation=data_Simulation.reset_index()
fig = px.pie(data_Simulation, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
data_Simulation=data[data['Genre']=='Simulation']

data_NA_Sales=data_Simulation
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Simulation
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Simulation
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Fighting=data[data['Genre']=='Fighting']
data_Fighting=data_Fighting['Publisher'].value_counts()[:20]
data_Fighting = pd.DataFrame(data_Fighting)
data_Fighting=data_Fighting.reset_index()
fig = px.pie(data_Fighting, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

data_Adventure=data[data['Genre']=='Adventure']
data_Adventure=data_Adventure['Publisher'].value_counts()[:20]
data_Adventure = pd.DataFrame(data_Adventure)
data_Adventure=data_Adventure.reset_index()
fig = px.pie(data_Adventure, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

data_Strategy=data[data['Genre']=='Strategy']
data_Strategy=data_Strategy['Publisher'].value_counts()[:20]
data_Strategy = pd.DataFrame(data_Strategy)
data_Strategy=data_Strategy.reset_index()
fig = px.pie(data_Strategy, values='Publisher', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

data_Fighting=data[data['Genre']=='Fighting']

data_NA_Sales=data_Fighting
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Fighting
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Fighting
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Adventure=data[data['Genre']=='Adventure']

data_NA_Sales=data_Adventure
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Adventure
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Adventure
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()
data_Strategy=data[data['Genre']=='Strategy']

data_NA_Sales=data_Strategy
data_NA_Sales = pd.DataFrame(data_NA_Sales)
fig = px.histogram(data_NA_Sales, x='Publisher', y='NA_Sales', histfunc='avg')
fig.show()

data_EU_Sales=data_Strategy
data_EU_Sales = pd.DataFrame(data_EU_Sales)
fig = px.histogram(data_EU_Sales, x='Publisher', y='EU_Sales', histfunc='avg')
fig.show()

data_JP_Sales=data_Strategy
data_JP_Sales = pd.DataFrame(data_JP_Sales)
fig = px.histogram(data_JP_Sales, x='Publisher', y='JP_Sales', histfunc='avg')
fig.show()