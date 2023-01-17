# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# here is the data 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_beers = pd.read_csv('../input/beers.csv')
data_brew = pd.read_csv('../input/breweries.csv')

# The first columns of breweries doesn't have a label, so add the label brewery_id to it. 
data_brew.columns = ['brewery_id', 'name', 'city', 'state']
#Merge the two different data files into one using brewery_id as the merge on key
merge_data = pd.merge(data_beers, data_brew, on=['brewery_id'])


#Pre-processing the data to handle the NaNs in the ibu columns 
from sklearn.preprocessing import Imputer 
imputer = Imputer()
#Imputer is meant to replace the NaNs in ibu column

##Get state name 
df_state = pd.DataFrame(merge_data['state'])

##Data frame for ibu an
values = merge_data['ibu'].values
values= values.reshape(-1,1)
ibu = imputer.fit_transform(values)

df_ibu = pd.DataFrame(ibu)

#concatenate State and ibu 
frames= [df_state, df_ibu]
state_ibu = pd.concat(frames, axis = 1)
state_ibu.columns = ['state', 'ibu']
# print(state_ibu[100:120])

# Group the values by states and take the mean of the values 
state_mean = state_ibu.groupby('state')['ibu'].agg(['count','mean']).reset_index()
# state_mean.head()
state_mean.columns = ['state', "# of breweries", 'mean IBU in beer']

print('Below are the top 5 states with the most breweries in them \n\n')
print(state_mean.sort_values(by=['# of breweries'], ascending = False).head())

print('\n\nBelow are the top 5 states with the the highest IBU in their beer \n\n')
print(state_mean.sort_values(by=['mean IBU in beer'], ascending = False).head())

# MAKE MAPS
import plotly
import plotly.plotly as py

plotly.tools.set_credentials_file(username='sanster9292', api_key='mYRVeX8qJoFou3GosOzU')

#for col in state_mean.columns:
#    state_mean[col] = state_mean[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
state_mean['text']= 'Breweries ='+ state_mean['count'].astype(str)+'<br>'+\
    'Mean IBU='+state_mean['mean'].astype(str)

data= [dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_mean['state'],
        z = state_mean['mean'].astype(float),
        locationmode = 'USA-states',
        text = state_mean['text'],
        marker = dict(
                line = dict(
                     color = 'rgb(255,255,255)',
                     width = 2
                        )),
        colorbar = dict(title ='ibu spread')
        )]

layout= dict(
            title ='How strong the beer of your state',
            geo=dict(
                    scope='usa',
                    projection=dict( type='albers usa'),
                    showlakes = True, 
                    lakecolor = 'rgb(255, 255, 255)'),
                    )
     
fig = dict( data=data, layout=layout )
py.iplot(fig, filename='d3-cloropleth-map')