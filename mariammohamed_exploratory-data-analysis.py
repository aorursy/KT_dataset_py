import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
drugsComTrain_raw = pd.read_csv('../input/drugsComTrain_raw.csv', parse_dates=['date'])
drugsComTrain_raw.shape
drugsComTrain_raw.head()
drugsComTrain_raw.describe()
len(drugsComTrain_raw.drugName.unique())
len(drugsComTrain_raw.condition.unique())
drugsComTrain_raw.rating.value_counts()
len(drugsComTrain_raw.date.unique())
drugsComTrain_raw.groupby(['drugName'])['rating'].agg(['count', 'mean']).sort_values(by=['count', 'mean'], ascending=False)[:10]
temp_df = drugsComTrain_raw.groupby(['drugName'])['rating'].agg(['count', 'mean']).sort_values(by=['count', 'mean'], ascending=False)
temp_df.describe()
# https://stackoverflow.com/questions/1411199/what-is-a-better-way-to-sort-by-a-5-star-rating
# (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C
# * R = average for the movie (mean) = (Rating)
# * v = number of votes for the movie = (votes)
# * m = minimum votes required to be listed in the Top 250 (currently 1300)
# * C = the mean vote across the whole report (currently 6.8)

C = drugsComTrain_raw.rating.mean() 
m = temp_df['count'].mean()
tempdf = drugsComTrain_raw.groupby(['drugName']).agg({'rating': [np.average, np.count_nonzero]}).reset_index()
v = tempdf['rating']['count_nonzero']
R = tempdf['rating']['average']
tempdf['actual_rating'] = (v * R + m * C) / (v + m)
tempdf.sort_values(by=['actual_rating'], ascending=False, inplace=True)
tempdf.head()
tempdf2 = drugsComTrain_raw.groupby(['drugName', 'condition']).agg({'rating': [np.average, np.count_nonzero]}).reset_index()
v2 = tempdf2['rating']['count_nonzero']
R2 = tempdf2['rating']['average']
tempdf2['actual_rating'] = (v2 * R2 + m * C) / (v2 + m)
tempdf2.sort_values(by=['actual_rating'], ascending=False, inplace=True)
tempdf2.set_index(['condition', 'drugName'], inplace=True)
tempdf2.head(20)
# example
tempdf2.loc['mance Anxiety', 'Propranolol']['actual_rating'][0]

drugsComTrain_raw['positive'] = drugsComTrain_raw['rating']>5
tempdf3 = drugsComTrain_raw.groupby(['drugName']).agg({'rating': [np.average, np.count_nonzero, np.std], 'positive': [np.sum]}).reset_index()
tempdf3.head()
tempdf3.fillna(0, inplace=True) 

tempdf3.head()

tempdf3['quality_normal'] = tempdf3['rating']['average']\
                - (1.96 * tempdf3['rating']['std'] / np.sqrt(tempdf3['rating']['count_nonzero']))

tempdf3.head()

np.min(tempdf3['quality_normal']), np.max(tempdf3['quality_normal'])

tempdf3['percentage'] = tempdf3['positive']['sum'] / tempdf3['rating']['count_nonzero']

tempdf3.head()

tempdf3['quality_binormal'] = tempdf3['percentage'] +   (1.96**2)/ (2*tempdf3['rating']['count_nonzero']) \
                - 1.96 * (np.sqrt(tempdf3['percentage'] * (1-tempdf3['percentage']) + 1.96**2 / (4*tempdf3['rating']['count_nonzero'])) / tempdf3['rating']['count_nonzero']) / \
                    (1 + 1.96**2 / tempdf3['rating']['count_nonzero'])

tempdf3.head()

np.min(tempdf3['quality_binormal']), np.max(tempdf3['quality_binormal'])


