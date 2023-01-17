import os

import pandas as pd
sbuxlocs = pd.read_csv('../input/store-locations/directory.csv', encoding = "ISO-8859-1")

sbuxlocs.columns = [c.replace(' ', '_') for c in sbuxlocs.columns]
# Eliminate duplicate stores if they exist

s1 = sbuxlocs.groupby('City').Store_Number.nunique().reset_index(name='counts')

s1.set_index('City',inplace=True)
cities = pd.read_csv('../input/cities-of-the-world/cities15000.csv', encoding = "ISO-8859-1")

cities.columns = [c.replace(' ', '_') for c in cities.columns]
# Coerce in case of invalid parsing

cities.population = pd.to_numeric(cities.population, errors='coerce')
# Eliminate duplicate cities by taking average of duplicate city populations if any

s2 = cities.groupby('name').population.mean().reset_index(name='population')

s2 = s2.rename(columns={'name': 'City'})

s2.set_index('City',inplace=True)
s3 = pd.concat([s1,s2],axis=1)

s3['count_per_capita'] = s3.counts/s3.population

s3.count_per_capita.sort_values(ascending=True)