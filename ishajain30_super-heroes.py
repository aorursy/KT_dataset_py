# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
from plotly import tools
py.init_notebook_mode(connected=True)
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
info = pd.read_csv('../input/heroes_information.csv')
powers = pd.read_csv('../input/super_hero_powers.csv')
info.head()
#preprocessing
info.isnull().sum()
info['Publisher'].value_counts()
#treat missing values
info.replace(to_replace='-', value = 'Other',inplace=True)
info['Publisher'].fillna('Other',inplace=True)
#drop column 'Unnamed'
info.drop('Unnamed: 0',axis=1,inplace=True)
info.head()
info.groupby('Publisher')['name'].count().sort_values(ascending=False)
sns.countplot(x='Gender', data = info)
sns.countplot(x='Alignment', data=info, hue='Gender')
#which publisher has more good v/s bad heroes
alignment_publisher = info[['Publisher', 'Alignment']]
alignment_publisher.head()
fig, ax = plt.subplots()

fig.set_size_inches(18,10)
sns.countplot(x=alignment_publisher['Publisher'], data=alignment_publisher, hue='Alignment')
powers.head()
#convert True to 1 and false to 0
power = powers*1
power.head()
power.loc[:, 'no_of_powers'] = power.iloc[:, 1:].sum(axis=1)
powerful_hero=power[['hero_names','no_of_powers']]
powerful_hero = powerful_hero.sort_values(by='no_of_powers', ascending=False)
powerful_hero
#plot top 10 superheroes
fig, ax = plt.subplots()

fig.set_size_inches(14,10.8)
sns.barplot(x=powerful_hero['hero_names'].head(10), y=powerful_hero['no_of_powers'].head(10), data=powerful_hero)
#height for top superheroes
newdata = info.merge(powerful_hero, how = 'inner', left_on='name', right_on='hero_names' )
newdata.drop('hero_names', axis=1, inplace=True)
newdata.head()
newdata['Height'].max()

newdata[newdata['Height']==975]
newdata['Height'].min()
newdata[newdata['Height']==-99]
#height based on number of powers
height_power = newdata[['name','Height', 'no_of_powers']]
sorted_height = height_power.sort_values(by='no_of_powers', ascending=False)
sorted_height.plot(x='no_of_powers', y='Height', kind='line')
newdata['Weight'].max()
newdata[newdata['Weight']==900]
#height based on number of powers
weight_power = newdata[['name','Weight', 'no_of_powers']]
sorted_weight = weight_power.sort_values(by='no_of_powers', ascending=False)
sorted_weight.head()
sorted_weight.plot(x='no_of_powers', y='Weight', kind='line')
#explore marvel comics and dc comics data
marvel_data = newdata[newdata['Publisher']=='Marvel Comics']
marvel_data.head()
dc_data = newdata[newdata['Publisher']=='DC Comics']
dc_data.head()
#gender distribution withing Marvel Comics
gender_series = marvel_data['Gender'].value_counts()
gender = list(gender_series.index)
gender_percentage = list((gender_series/gender_series.sum())*100)

dc_gender_series = dc_data['Gender'].value_counts()
dc_gender = list(dc_gender_series.index)
dc_distribution = list((dc_gender_series/dc_gender_series.sum())*100)


fig = {
    'data': [
        {
            'labels': gender,
            'values': gender_percentage,
            'type': 'pie',
            'name': 'marvel gender distribution',
            'domain': {'x': [0, .48],
                       'y': [0, .49]},
            'hoverinfo':'percent+name',
            'textinfo':'label'
            
        },
        {
            'labels': dc_gender,
            'values': dc_distribution,
            'type': 'pie',
            'name': 'DC gender distribution',
            'domain': {'x': [.52, 1],
                       'y': [0, .49]},
            'hoverinfo':'percent+name',
            'textinfo':'label'

        },
       
    ],
    'layout': {'title': 'Comics gender distribution',
               'showlegend': True}
}

py.iplot(fig, filename='pie_chart_subplots')

dc_data['Gender'].value_counts()
gender_series
