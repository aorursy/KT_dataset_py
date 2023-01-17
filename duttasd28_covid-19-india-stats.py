from IPython.display import Image

Image('https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/header_1.png', width = 1920)
!pip -q --disable-pip-version-check install mplcyberpunk
## Necessary imports of libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import mplcyberpunk

%matplotlib inline
# Age group Data

ageGroupCases = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv', index_col = 'Sno')



# fuction to convert

def fromPercentToNumber(value):

    # Converts x.yz% string to float x.yz

    value = value.rstrip('%')

    return float(value)



# read data

ageGroupCases['Percentage'] = ageGroupCases['Percentage'].apply(fromPercentToNumber)



plt.figure(figsize = (8, 8))

plt.title('Age Wise Covid patients')

sns.barplot(data = ageGroupCases, x = 'AgeGroup', y = 'Percentage', palette='icefire');
covidData = pd.read_csv('../input/covid19-in-india/covid_19_india.csv',

                        index_col = 'Sno',

                        parse_dates=['Date'])

covidData.drop(['Time'], axis = 1, inplace = True)
plt.figure(figsize = (25, 5))

## Data for Kerala

plt.title('COVID Confirmed cases for Kerala from January till Now')

plt.style.use('cyberpunk')

sns.lineplot(data = covidData[covidData['State/UnionTerritory'] == 'Kerala'],

             x = 'Date',

             y = 'Confirmed',

            color = 'magenta');

mplcyberpunk.add_glow_effects()
plt.figure(figsize = (25, 5))

## Data for West Bengal

plt.title('COVID Confirmed cases for West Bengal from January till Now')

plt.style.use('dark_background')

sns.lineplot(data = covidData[covidData['State/UnionTerritory'] == 'West Bengal'],

             x = 'Date',

             y = 'Confirmed',

            color = 'yellow');
plt.figure(figsize = (25, 5))

## Data for Rajasthan

plt.title('COVID Confirmed cases for Rajasthan from January till Now')

sns.set(style = "whitegrid")

sns.lineplot(data = covidData[covidData['State/UnionTerritory'] == 'Rajasthan'],

             x = 'Date',

             y = 'Confirmed',

            color = 'blue');
delhiStats = covidData[covidData['State/UnionTerritory'] == 'Delhi']

sns.set(style = "dark")

plt.figure(figsize = (25, 5))

plt.title("Delhi Deaths per Day Basis")

fig = sns.barplot(y = 'Deaths', x = 'Date', data = delhiStats, palette = 'winter')

plt.xticks([])

plt.show(fig)
plt.figure(figsize = (25, 5))

sns.set(style = 'white')

fig = sns.barplot(y = delhiStats['Deaths'].cumsum(), x = 'Date', data = delhiStats, palette = 'autumn')

plt.title("Delhi Deaths(Cumulative)")

plt.xticks([])

plt.show(fig)