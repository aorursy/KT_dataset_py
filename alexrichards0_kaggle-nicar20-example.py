import pandas as pd

import numpy as np

import seaborn

import matplotlib.pyplot as plt

import requests
!pip install pyreadr
!wget -O nola_stops_wget.rds https://github.com/richardsalex/data_examples/raw/master/nola_policing_data.rds
# Use pyreadr package to convert RDS to a dataframe

import pyreadr

nola_stops = pyreadr.read_r("nola_stops_wget.rds")

df = nola_stops[None]
# Take a look to see what we're working with

df.head()
# Convert 'date' to correct data type

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

# Annual NOPD stops by race/ethnicity

df.groupby([pd.DatetimeIndex(df['date']).year, 'subject_race']).size()
sf_rates = df[['subject_race', 'search_conducted', 'frisk_performed']].groupby('subject_race').mean().reset_index()



ax = seaborn.catplot(x="subject_race", y="search_conducted", kind="bar", palette='Pastel1', aspect=1.8, data=sf_rates)

plt.show()
# Let's try this again, breaking it all down by year as well

df['year'] = pd.DatetimeIndex(df['date']).year

ann_rates = df[['year', 'subject_race', 'search_conducted', 'frisk_performed']].groupby(['year','subject_race']).mean().reset_index()

ann_rates
plt.figure(figsize=(11,6))

ax = seaborn.lineplot(x="year", y="frisk_performed", hue="subject_race", style="subject_race", markers=True, dashes=False, data=ann_rates)

ax.set(xlabel="Year", ylabel="Frisk Performed")

plt.show()