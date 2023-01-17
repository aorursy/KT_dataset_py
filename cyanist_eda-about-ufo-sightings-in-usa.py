import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns



import plotly.plotly as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt



from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



plt.rcParams['figure.figsize'] = (12, 6)
df0 = pd.read_csv('../input/complete.csv', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], low_memory=False)
# check if there are missing values in dataset

df0.isnull().values.any()
# count NaN values in each column

df0.isnull().sum()
# fill NaN values with '0'

df=df0.fillna(value=0)

df
# check all columns' data types

df.dtypes
# change type of latitude to float

df['latitude'] = pd.to_numeric(df['latitude'],errors='coerce')

df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'],errors='coerce')
df.dtypes
# make city, state, country columns more pretty

df['city']=df['city'].str.title()

df['state']=df['state'].str.upper()

df['country']=df['country'].str.upper()
# check if there are inappropraite values in dataset

df.describe().astype(np.int64).T
# replace inappropraite values with column mean

df.replace([97836000,0],df['duration (seconds)'].mean())
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# df.insert(1, 'year', df['datetime'].dt.year)

df['year'] = df['datetime'].dt.year

df['hour'] = df['datetime'].dt.hour

df['year'] = df['year'].fillna(0).astype(int)

df['hour'] = df['hour'].fillna(0).astype(int)

df['city'] = df['city'].str.title()

df['state'] = df['state'].str.upper()

df['country'] = df['country'].str.upper()

df['shape'] = df['shape'].str.title()

df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')



us_states = np.asarray(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',

                        'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',

                        'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',

                        'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',

                        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
df.describe()
df.info()
# count each state's ufo sighting times

df["state"].value_counts()
# show the current information of dataset

df.info()
sns.distplot(df['year'])

plt.xlim(1900,2015)

plt.show()
sns.distplot(df['hour'])

plt.show()
countries = df['country']

country_count = countries.value_counts()

country_count[:5].plot(kind='bar')

plt.show()
# draw the barplot of the first 20 states' ufo sightings times

states = df['state']

state_count = states.value_counts()

state_count[:51].plot(kind='bar')

plt.show()
# draw the barplot of different shapes' ufo sightings times

shapes = df['shape']

shape_count = shapes.value_counts()

shape_count[:15].plot(kind='bar')

plt.show()
# find the summary statistics for each column

df.describe(include='all')
sns.boxplot(x="shape", y="longitude", data=df)

plt.show()
sns.stripplot(x="shape", y="longitude", data=df, jitter=True)

plt.show()
sns.boxplot(x="shape", y="latitude", data=df)

plt.show()
sns.stripplot(x="shape", y="latitude", data=df, jitter=True)

plt.show()
sns.countplot(x="state", data=df, palette="Greens_d")

plt.show()
sns.countplot(x="shape", data=df, palette="Greens_d")

plt.show()
sns.regplot(x="year", y="latitude", data=df)

plt.xlim(1960,2015)

plt.show()
sns.lmplot(x="longitude", y="latitude", data=df)

plt.show()