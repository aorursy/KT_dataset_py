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
raw_data = pd.read_csv("../input/covid_19_data.csv")



#display(raw_data.shape)

display(raw_data.head())





# data cleaning

data = pd.DataFrame(raw_data, columns = ['ObservationDate', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered'])

data['Active'] = data['Confirmed'] - (data['Deaths'] + data['Recovered'])

data['Country/Region'] = data['Country/Region'].replace('Mainland China', 'China')

data['Country/Region'] = data['Country/Region'].replace('Iran (Islamic Republic of)', 'Iran')

	

data = data.fillna(0)



#display(data.head())



temp = data.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

#print(data.groupby('ObservationDate')['Deaths'].sum().max())

display(temp.max())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



value = data.groupby('ObservationDate')['Confirmed'].sum().max()

value = np.append(value, data.groupby('ObservationDate')['Recovered'].sum().max())

value = np.append(value, data.groupby('ObservationDate')['Active'].sum().max())

value = np.append(value, data.groupby('ObservationDate')['Deaths'].sum().max())



label = ['Confirmed', 'Recoverd', 'Active', 'Deaths']



print(value)



y_pos = np.arange(len(label))

 

# Create bars

plt.bar(y_pos, value, width=0.8)

 

# Create names on the x-axis

plt.xticks(y_pos, label)

 

# Show graphic

plt.show()
grouped = data.groupby(['Country/Region'])['Confirmed'].max().reset_index()

#display(grouped.head(10))



temp = grouped.sort_values('Confirmed', ascending=False).reset_index(drop=True)

display(temp.head(10))
grouped = data.groupby(['Country/Region'])['Deaths'].max().reset_index()

#display(grouped.head(10))



temp = grouped.sort_values('Deaths', ascending=False).reset_index(drop=True)

display(temp.head(10))