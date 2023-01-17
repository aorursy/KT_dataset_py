# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context('paper')

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def random_colours(number_of_colors):

    '''

    Simple function for random colours generation.

    Input:

        number_of_colors - integer value indicating the number of colours which are going to be generated.

    Output:

        Color in the following format: ['#E86DA4'] .

    '''

    colors = []

    for i in range(number_of_colors):

        colors.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))

    return colors
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.head()
data.info()
data.describe()
data.describe(include="O")
data['Last Update'] = pd.to_datetime(data['Last Update'])
data['Day'] = data['Last Update'].apply(lambda x:x.day)

data['Hour'] = data['Last Update'].apply(lambda x:x.hour)
data[data['Day'] == 30]
data[data['Day'] == 30].sum()
latest_data = data[data['Day'] == 30]
latest_data.head()
print('Globally Confirmed Cases: ',latest_data['Confirmed'].sum())

print('Global Deaths: ',latest_data['Deaths'].sum())

print('Globally Recovered Cases: ',latest_data['Recovered'].sum())
plt.figure(figsize=(16,6))

data.groupby('Day').sum()['Confirmed'].plot()
plt.figure(figsize=(16,6))

sns.barplot(x='Day',y='Confirmed',data=data)
latest_data.groupby('Country').sum()
data[data['Confirmed']==0]
data = data[data['Confirmed'] != 0]
plt.figure(figsize=(18,8))

sns.barplot(x='Country',y='Confirmed',data=data)

plt.tight_layout()
import plotly.express as px

fig = px.bar(data, x='Province/State', y='Confirmed')

fig.show()
plt.figure(figsize=(16,6))

temp = latest_data.groupby(['Province/State']).sum()['Confirmed'].plot.bar()
pivoted = pd.pivot_table(data, values='Confirmed', columns='Country', index='Day')

pivoted.plot(figsize=(16,10))
pivoted = pd.pivot_table(data, values='Confirmed', columns='Province/State', index='Day')

pivoted.plot(figsize=(20,15))
data[data['Day'] == 22]['Country'].unique()
temp = data[data['Day'] == 22]

temp.groupby('Country').sum()['Confirmed'].plot.bar()
data[data['Day'] == 30]['Country'].unique()
plt.figure(figsize=(16,6))

temp = data[data['Day'] == 30]

temp.groupby('Country').sum()['Confirmed'].plot.bar()
data_main_china = latest_data[latest_data['Country']=='Mainland China']
(data_main_china['Deaths'].sum() / data_main_china['Confirmed'].sum())*100
(data_main_china['Recovered'].sum() / data_main_china['Confirmed'].sum())*100
data_main_china.groupby('Province/State')['Deaths'].sum().reset_index().sort_values(by=['Deaths'],ascending=False).head()
plt.figure(figsize=(16,6))

data.groupby('Day').sum()['Deaths'].plot()
pivoted = pd.pivot_table(data[data['Country']=='Mainland China'] , values='Confirmed', columns='Province/State', index='Day')

pivoted.plot(figsize=(20,15))
pivoted = pd.pivot_table(data, values='Deaths', columns='Province/State', index='Day')

pivoted.plot(figsize=(20,15))