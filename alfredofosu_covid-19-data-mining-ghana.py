# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # mathematical operations and linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library

import seaborn as sns # Fancier visualizations

import statistics # fundamental stats package

%matplotlib inline

import os

import scipy.stats as stats # to calculate chi-square test stat

from datetime import date

import plotly.graph_objects as go

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load raw data from url and use first row as column names



rawData = pd.read_csv('../input/jhu-csse-rawdata/coronavirus.csv', sep=',',

                           header=0, encoding='ascii', engine='python')

df = rawData
df
df.sort_values(['country','type','date'],inplace = True)

df.reset_index(drop=True, inplace=True) # reset index from 0 to -1

print(df)
df.info() #Display Data Types and Columns
df['date'] = df['date'].astype('datetime64')

df = df.fillna('unknown')

df.info()
df.columns # get column names 
df_conf_ttl = df[df.type == 'confirmed'].cases.sum()

df_deat_ttl = df[df.type == 'death'].cases.sum()

df_rcvd_ttl = df[df.type == 'recovered'].cases.sum()

ObservationDate = df['date'].max() #Latest date

df_ac_ttl = df_conf_ttl  - (df_deat_ttl + df_rcvd_ttl) # Active cases



labels = ["Last Update","Confirmed","Active cases","Recovered","Deaths"]

fig = go.Figure(data=[go.Table(header=dict(values=labels),

                 cells=dict(values=[ObservationDate,df_conf_ttl,df_ac_ttl,df_rcvd_ttl,df_deat_ttl]))

                     ])

fig.update_layout(

    title='Total Number of COVID 19 Cases in The world',

)

fig.show()
df['date'].max()-df['date'].min() # number of tracked day data has been tracked
dfghana = df[df.country == 'Ghana']

dfghana = dfghana.reset_index(drop=True, inplace=None) #reduce to 74 x 4

dfghana = dfghana[['date','type','cases']]

dfghana
# Confirmed Cases in Ghana

dfghana_conf_ttl = dfghana[dfghana.type =='confirmed']

dfghana_conf_ttl = dfghana_conf_ttl[['date','cases']]

dfghana_conf_ttl
# Death Cases in Ghana

dfghana_deat_ttl = dfghana[dfghana.type =='death']

dfghana_deat_ttl = dfghana_deat_ttl[['date','cases']]

dfghana_deat_ttl
# Recovered Cases in Ghana

dfghana_rcvd_ttl = dfghana[dfghana.type =='recovered']

dfghana_rcvd_ttl =  dfghana_rcvd_ttl[['date','cases']]

dfghana_rcvd_ttl
#Actice Cases in Ghana

dfghana_actv_ttl = (dfghana_conf_ttl['cases'].sum() - (dfghana_deat_ttl['cases'].sum() 

                    + dfghana_rcvd_ttl['cases'].sum()))



dfghana_actv_ttl
dfghana_merged = pd.merge(dfghana_conf_ttl,dfghana_deat_ttl, on = 'date', how = 'right')

dfghana_merged = pd.merge(dfghana_merged,dfghana_rcvd_ttl, on = 'date', how = 'right')

dfghana_merged
#rename columns

dfghana_merged = dfghana_merged.rename(columns ={'cases_x':'confirmed',

                                          'cases_y': 'death',

                                          'cases':'recovered'})
dfghana_merged
dfghana_merged.describe()
#Frequency and five number summary boxplot

dfghana_merged[['confirmed', 'death', 'recovered']].hist(layout=(1,3), sharex=False, sharey=False, figsize=(15, 5), bins=20) 

plt.show()



dfghana_merged[['confirmed', 'recovered', 'death']].plot(kind = 'box',subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(15,5))

plt.show()
print('-----------Skewness-------------')

print(dfghana_merged.skew(axis = 0, skipna = True))

print('\n-----------Kurtosis-------------')

print(dfghana_merged.kurtosis(skipna = True))
fig = go.Figure()

fig.add_trace(go.Scatter(x=dfghana_merged['date'],y=dfghana_merged['confirmed'],

             mode='lines+markers',

             name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=dfghana_merged['date'],y=dfghana_merged['death'],

             mode='lines+markers',

             name='Death Cases'))

fig.add_trace(go.Scatter(x=dfghana_merged['date'],y=dfghana_merged['recovered'],

             mode='lines+markers',

             name='Recovery Cases'))



fig.update_xaxes(

    rangeslider_visible=True

)



fig.show()