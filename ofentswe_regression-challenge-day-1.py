# library we'll need

import pandas as pd

import seaborn as sns

sns.set(color_codes=True)



# read in all three datasets (you'll pick one to use later)

recpies = pd.read_csv("../input/epirecipes/epi_r.csv", engine='python')

bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv", engine='python')

weather = pd.read_csv("../input/szeged-weather/weatherHistory.csv", engine='python')
# quickly clean our dataset

recpies = recpies[recpies.calories < 10000]

recpies = recpies.dropna()
# are the ratings all numeric?

print("Is this variable numeric?")

pd.api.types.is_numeric_dtype(recpies.rating)
# are the ratings all integers?

print("Is this variable only integers?")

recpies.rating.equals(pd.to_numeric(recpies.rating)) == True
# plot calories by whether or not it's a dessert

sns.lmplot(x='calories', y='dessert', data=recpies, fit_reg=False, 

           scatter_kws={"marker": "D", "s": 100,"color": "black"}, size=12, x_jitter=.1)
# plot & add a regression line

sns.lmplot(x='calories', y='dessert', data=recpies,

           scatter_kws={"marker": "D", "s": 100,"color": "black"}, size=14, x_jitter=.1)
# your work goes here! :)

weather.head()
weather = weather.dropna()
sns.lmplot(x='Temperature (C)', y='Apparent Temperature (C)', data=weather, fit_reg=False, 

           scatter_kws={"marker": "D", "s": 100,"color": "black"}, size=12, x_jitter=.1)
sns.lmplot(x='Temperature (C)', y='Apparent Temperature (C)', data=weather,

           scatter_kws={"marker": "D", "s": 100,"color": "black"}, size=14, x_jitter=.1)