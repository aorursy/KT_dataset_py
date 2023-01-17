# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py #heat map by state tools

import plotly.tools as tls



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/states_all.csv')
df.isna().sum()
corr = df.corr()

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(corr, annot=True,

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values)
fig, ax = plt.subplots(figsize=(15,15))

sns.violinplot(x="TOTAL_REVENUE", y="STATE", data=df)

ax.set_title('Total Revenue by State')
df['average_revenue']=df['TOTAL_REVENUE']/df['GRADES_ALL_G']

df['average_expenditure']=df['TOTAL_EXPENDITURE']/df['GRADES_ALL_G']
fig, ax = plt.subplots(figsize=(15,15))

sns.violinplot(x="average_revenue", y="STATE", data=df[df.STATE!="Virgina"])
df.head()
states = pd.read_csv('../input/states_all.csv')

states.head()
states['text'] = 'TOTAL REVENUE'+states['TOTAL_REVENUE'].astype(str) + 'Total Expenditure'+states['TOTAL_EXPENDITURE'].astype(str) + 'State '+states['STATE']



data = [dict(type='choropleth',autocolorscale=False, locations = states['STATE'], z= states['TOTAL_REVENUE'], locationmode='USA-states',

                text = states['text'], colorscale = 'Blues', colorbar= dict(title="Revenue by Expenditure by State"))]

data
layout = dict(title='State Expenditure vs Revenue', 

              geo = dict(scope='usa', projection=dict(type='albers usa'), showlakes = True, lakecolor = 'rgb(66,165,245)',),)

layout
fig = dict(data=data, layout=layout)



py.iplot(fig, filename="State_expnd_vs_incm")