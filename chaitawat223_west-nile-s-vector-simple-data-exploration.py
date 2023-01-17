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
df = pd.read_csv('../input/west-nile-virus-wnv-mosquito-test-results.csv',parse_dates=True)
df['datetime'] = pd.to_datetime(df['TEST DATE'])
df.set_index('datetime',inplace=True)
df.sort_index(inplace=True)
df.head(10)
#This dataset contains 11 years of West Neil's vetor monitoring data
#Let's find out top 3 where positive vectors were found the most in 11 years

#First, let's convert "RESULT" to integer
label = {'positive':1,'negative':0}
df['RESULT'] = df['RESULT'].map(label)
df['RESULT'] = df['RESULT'].astype('int')

#Select only result containing positive vector
pos = df[df['RESULT'] == 1]
pos.head(10) #display only positive vectors
#Now visualize the amount of positive vectors
import matplotlib.pyplot as plt

plt.figure(figsize=(10,9))

#pos_mos sums positive vectors(1) by year
#It represents number of positive vectors found in each year
pos_mos = pos.groupby(pos['SEASON YEAR']).sum()['RESULT'].plot.bar()
pos_mos.set_ylabel('Amount of positive vector')

#Add text above each bar
#Took me little long to find this method!!!!
#pos_mos.patches is a way to use property in site the object
for i in pos_mos.patches:
    x_value = i.get_x()
    y_value = i.get_height()
    plt.annotate(str(y_value),xy=(x_value+0.05,y_value+3))
    
#Now we figure out where should we focus on
#Ranked from highest amount year 2016,2012,2013,and 2007 show highest positive vectors
#Based on above, let's see what species were found the highest to be positive
#I wonder how each species spread in 11 year
pos_hy = pos[pos['SPECIES'] == 'CULEX PIPIENS/RESTUANS']
pos_pi = pos[pos['SPECIES'] == 'CULEX PIPIENS']
pos_re = pos[pos['SPECIES'] == 'CULEX RESTUANS']

#Visualize them
plt.figure(figsize=(10,9))
pos_hy['SPECIES'].groupby(pos_hy.index.year).count().plot()
pos_pi['SPECIES'].groupby(pos_pi.index.year).count().plot()
pos_re['SPECIES'].groupby(pos_re.index.year).count().plot()
plt.legend(['CULEX PIPIENS/RESTUANS','CULEX PIPIENS','CULEX RESTUANS'])
plt.title('Distribution of species')
#Now select only 2016,2012,2013,and 2007 based on the highest positive vectors
pos_new = pos[(pos.index.year == 2016) | (pos.index.year == 2012) | (pos.index.year == 2013) | (pos.index.year == 2007)]
pos_new.head(10)
#Looks like Culex pipines/restuans is found the most among the selected 4 years
pos_new['SPECIES'].groupby(pos_new['SEASON YEAR']).value_counts()
#What about the current year? what species overwhelm 2018
#OK!!! this year, we should consider to contain 2 following species
#In addition, refer back to line graph above, we could consider these 2 species a serious problem.
#We need to contain this, but where do we find them?

pos[pos.index.year == 2018]['SPECIES'].value_counts().plot(kind='bar')
#Based on description, the monitoring took place in Chicago, USA
#We could narrow down our map according to information
#However,it should be better to get rid of blank field of location

#Here is what we do not need 588 records with missing data
pos[pos['LATITUDE'].isna() & pos['LONGITUDE'].isna()].shape
#Ok, now we could identify missing data
pos['check_empty'] = pos[['LATITUDE','LONGITUDE']].apply(lambda x: 0 if x.isna().all() else 1,axis=1)
pos['check_empty'].value_counts()
pos_map = pos[pos['check_empty'] == 1] #here is our new dataframe without missing data
pos_map = pos_map[pos_map.index.year == 2018] #we use data from 2018

pos_map_hy = pos_map[pos_map['SPECIES'] == 'CULEX PIPIENS/RESTUANS'] #hybride species
pos_map_re = pos_map[pos_map['SPECIES'] == 'CULEX RESTUANS'] # C. restuans species
#It looks like these 2 species strat increaseing from the middle to the end of year 2018
#Wet climate, I guess
#Consistant to previous line graph, the amount of the vectors tend to raise at the of every year.

plt.figure(figsize=(10,9))
pos_map_hy['SPECIES'].groupby(pos_map_hy.index).count().plot()
pos_map_re['SPECIES'].groupby(pos_map_re.index).count().plot()
plt.legend(['CULEX PIPIENS/RESTUANS','CULEX RESTUANS'])
plt.title('Distribution of species in 2018')
#This part is a real challeng in visulaization, ploting vectors'distribution
#Took me long to learn basic "Plotly". This library is useful for interactive visualization and making dash board
#Before applying this library,i suggest you guys to creat account in "Plotly" for accessing library and "Mapbox" to access base map offline


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) #Offline mode

#this is public token form mapbox which you can create your own in "Mapbox", no charge!!!!!
token = 'pk.eyJ1IjoiY2hhaXRhd2F0IiwiYSI6ImNqbHE0NGluZTA4N3EzcHRhMGZlejV2Y3YifQ.7yqd7CJ_7qVhxEfJLtN3VQ'

data_hy = [go.Scattermapbox(lon=pos_map_hy['LONGITUDE'],lat=pos_map_hy['LATITUDE'],text=pos_map_hy['SPECIES'],mode='markers',marker=dict(size=10)),
          go.Scattermapbox(lon=pos_map_re['LONGITUDE'],lat=pos_map_re['LATITUDE'],text=pos_map_re['SPECIES'],mode='markers',marker=dict(size=10))]


layer_hy = go.Layout(title = "CULEX PIPIENS/RESTUANS and RESTUANS's location 2018",hovermode='closest',autosize=True,mapbox=dict(accesstoken=token))

fig = dict(data=data_hy,layout=layer_hy)
iplot(fig)