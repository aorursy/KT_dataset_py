import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



results = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')

results.head(10)
def find_winners(df):

    winners = []

    for i, row in df.iterrows():

        if row['home_score'] > row['away_score']:

            winners.append(row['home_team'])

        elif row['home_score'] < row['away_score']:

            winners.append(row['away_team'])

        else:

            winners.append('Draw')

    return winners

results['winner'] = find_winners(results)

results.head()
def find_team_win(df):

    win = []

    for i, row in df.iterrows():

        if row['home_score'] > row['away_score']:

            win.append('home_team')

        elif row['home_score'] < row['away_score']:

            win.append('away_team')

        else:

            win.append('Draw')

    return win

results['home_away_win'] = find_team_win(results)
results["home_subtract_away_score_diff"] = results["home_score"]-results["away_score"]
neutral = []

for index, row in results.iterrows():

    neutral.append((row['country'] not in  row['home_team']) and (row['home_team'] != 'USA'))

    

results['neutral'] = neutral

results['neutral'] = results['neutral'].astype(int)



results.head()
results.isnull().sum()
cities = pd.read_csv('../input/world-cities-database/worldcitiespop.csv', encoding='latin-1', dtype={

    'Country': str,

    'City': str,

    'AccentCity': str,

    'Region': str

})

cities = cities.dropna()

cities = cities[['Country', 'AccentCity', 'Latitude', 'Longitude']]

cities.head()
countries = pd.read_csv('../input/world-countries-and-continents-details/countries and continents.csv')

countries = countries.rename(columns = {'official_name_en': 'Name', 'ISO3166-1-Alpha-2': 'Code'})

countries = countries[['Name', 'Code']]

countries['Code'] = countries['Code'].str.lower()
coords = results[['city', 'country']]

df = coords.merge(cities, how='inner', left_on='city', right_on='AccentCity')

new_df = df.merge(countries, how='left', left_on='Country', right_on='Code')

city_coords = new_df[['city', 'Latitude', 'Longitude']]

city_coords = city_coords.drop_duplicates('city')

city_coords = city_coords.rename(columns={'Latitude':'latitude', 'Longitude': 'longitude'})
new_results = results.merge(city_coords, how='left', on='city')

new_results.head()
new_results.to_csv("international-football-results.csv.gz",compression="gzip",index=False)