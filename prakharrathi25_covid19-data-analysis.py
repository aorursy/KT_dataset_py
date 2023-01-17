# Data Analysis 

import numpy as np 

import pandas as pd 



# Data Visualisation 

import matplotlib.pyplot as plt 

import seaborn as sns 

print("Modules Imported")
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
# Check the current Directory

os.getcwd()
# Import Covid19 Data 

covid_data = pd.read_csv('../input/covid19/covid19_Confirmed_dataset.csv')

covid_data.shape
# Print the first 5 rows

covid_data.head()
# Check unique countries 

countries = list(covid_data['Country/Region'].unique())

print("Number of Countries: ", len(countries))
# Remove Lat, Long columns 

covid_data.drop(['Lat', 'Long'], axis=1, inplace=True)

covid_data.shape
# Group data by country 

covid_data_agg = covid_data.groupby('Country/Region').sum()

print(covid_data_agg.shape)
covid_data_agg.head()
# Find the data for a particular country and plot it

covid_data_agg.loc['China'].plot()

covid_data_agg.loc['Italy'].plot()

covid_data_agg.loc['India'].plot()

plt.legend()
# Plot for the first n days

n = 3

covid_data_agg.loc['China'][:n].plot()
covid_data_agg.loc['China'].diff().plot()
# Calculate the maximum and minimum increase in the cases

max_inc = covid_data_agg.loc['China'].diff().max()

min_inc = covid_data_agg.loc['China'].diff().min()



print("Maximum Increase in a day:", max_inc)

print("Minimum Increase in a day: ", min_inc)
# Make a list of max infections for each country

max_infections = []

for c in countries: 

    max_infections.append(covid_data_agg.loc[c].diff().max())



# Add to the data

covid_data_agg['max_infection_rate'] = max_infections
# Create a new dataframe 

df = pd.DataFrame(covid_data_agg['max_infection_rate'])

df.head()
# Load Happiness Report Data

happiness_data = pd.read_csv('../input/covid19/worldwide_happiness_report.csv')

happiness_data.shape
# Let's look at the first five rows 

happiness_data.head()
# List of useless columns 

useless_cols = ['Overall rank', 'Score', 'Generosity', 'Perceptions of corruption']



# Drop the useless cols 

happiness_data.drop(useless_cols, inplace=True, axis=1)

happiness_data.shape
happiness_data.set_index('Country or region', inplace=True)
data = df.join(happiness_data, how='inner')

data.head()
x = data['GDP per capita']

y = data['max_infection_rate']



# Make a scatter plot but due to the scale a logplot should be plotted

sns.scatterplot(x, np.log(y))
# Make a regression plot 



sns.regplot(x, np.log(y))
# Plot a correlation matrix and viualise it

sns.heatmap(data.corr(), annot=True)