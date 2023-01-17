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
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

world = pd.read_csv('/kaggle/input/world-coordinates/world_coordinates.csv')

df['Date'] = df['Date'].apply(pd.to_datetime)

df['Last Update'] = df['Last Update'].apply(pd.to_datetime)
d = df['Date'][-1:].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



from datetime import date

df_latest = df[df['Date'] > pd.Timestamp(date(year, month, day))]
df_latest['Last Update'] = pd.to_datetime(df_latest['Last Update'])

df_latest['Day'] = df_latest['Last Update'].apply(lambda x:x.day)

df_latest['Hour'] = df_latest['Last Update'].apply(lambda x:x.hour)



df_latest.head()
drop_element = ['Sno']

df_latest.drop(drop_element, axis=1, inplace = True)



#Merge Hongkong and Taiwan data in China

df_latest['Country'].replace({'Mainland China':'China', 

                              'Hong Kong':'China',

                             'Taiwan':'China'}, inplace = True)



print(df_latest.Country.unique())

print("\nThere are %s countries are affected during this periode" %format(len(df_latest.Country.unique())))
df_latest.groupby(['Country', 'Province/State']).sum()
df_latest.groupby('Country')['Deaths'].sum().sort_values(ascending = False)[:5]
df_latest.groupby('Country')['Recovered'].sum().sort_values(ascending = False)[:5]
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.graph_objects as go

from plotly.subplots import make_subplots
dfww = df_latest.groupby(['Country']).agg({

    'Confirmed':sum,

    'Deaths':sum,

    'Recovered':sum

}).reset_index().sort_values('Confirmed', ascending = False)



dfww.head()
trace1 = go.Bar(name = 'Confirmed', x = dfww['Country'][1:], y = dfww['Confirmed'][1:])

trace2 = go.Bar(name = 'Deaths', x = dfww['Country'][1:], y = dfww['Deaths'][1:])

trace3 = go.Bar(name = 'Recovered', x = dfww['Country'][1:], y = dfww['Recovered'][1:])

data = [trace1, trace2,trace3]



fig = go.Figure(data)

fig.update_layout(

    barmode = 'stack',

    title = 'Number of Confirmed/Recovered in the World'

    )





fig.show()
df['Date'] = df['Date'].dt.date

cond = df[df['Date'] > pd.Timestamp(date(2020, 1, 1))]

condgb = cond.groupby('Date').agg({'Confirmed':sum,

                                  'Deaths':sum,

                                  'Recovered':sum}).reset_index()

fig = make_subplots(rows = 1, cols = 3, subplot_titles = ("Confirmed","Deaths","Recovered"))



fig.add_trace(go.Scatter(name = 'Confirmed',

                         x = condgb['Date'],

                         y = condgb['Confirmed'],

                         opacity = 0.8),

              row = 1, col = 1)



fig.add_trace(go.Scatter(name ='Deaths',

                          x = condgb['Date'], 

                          y = condgb['Deaths'],

                          line_color = 'red',

                          opacity = 0.8),

               row = 1, col=2)

               



fig.add_trace(go.Scatter(name = 'Recovered',

                         x = condgb['Date'],

                         y = condgb['Recovered'],

                         line_color = 'Green',

                         opacity = 0.8),

              row = 1, col = 3)





fig.update_layout(title_text="Global infection of nCoV over time")



fig.show()
dfcn = df_latest[df_latest['Country'] == 'China'].sort_values('Confirmed', ascending = False)

dfcn.head()
print("In China, There are %s province/state are affected" %format(len(dfcn['Province/State'].unique())))
trace1 = go.Bar(name = 'Confirmed', x = dfcn['Province/State'][1:], y = dfcn['Confirmed'][1:])

trace2 = go.Bar(name = 'Recovered', x = dfcn['Province/State'][1:], y = dfcn['Recovered'][1:])



data = [trace1, trace2]

fig = go.Figure(data)

fig.update_layout(

    barmode = 'stack',

    title = 'Confirmed vs Recovered figures of Provinces of China other than Hubei'

)



fig.show()
df['Last Update'] = pd.to_datetime(df['Last Update'])

df['Day'] = df['Last Update'].apply(lambda x:x.day)

df['Hour'] = df['Last Update'].apply(lambda x:x.hour)
plt.figure(figsize = (16,12))

sns.set_style("whitegrid")

sns.lineplot(data = df[df['Province/State'] == 'Hubei']['Confirmed'], label = 'Hubei confirmed casses')

sns.lineplot(data = df[df['Province/State'] == 'Zhejiang']['Confirmed'], label = 'Zhejiang confirmed casses')

sns.lineplot(data = df[df['Province/State'] == 'Guangzhou']['Confirmed'], label = 'Guangzhou confirmed casses')



plt.show()