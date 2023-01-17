from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd

from pandas_profiling import ProfileReport



import seaborn as sns

import missingno as msno

from scipy import stats

sns.set(color_codes=True)

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/ipl-dataset-20082019/matches.csv", index_col=0)

df.head()
df.isnull().sum()
df.Season = df.Season.str.replace(r'IPL-', '').astype(int)

df.drop(columns=["umpire3"], inplace = True)
df.city = df.city.fillna("-")

df.umpire1 = df.umpire1.fillna("-")

df.umpire2 = df.umpire2.fillna("-")

df = df.replace('Rising Pune Supergiants', 'Rising Pune Supergiant')

df = df.replace('Pune Warriors', 'Rising Pune Supergiant')

df = df.replace('Deccan Chargers', 'Sunrisers Hyderabad')

df = df.replace('Delhi Capitals', 'Delhi Daredevils')
is_NaN = df.isnull()

row_has_NaN = is_NaN.any(axis=1)

rows_with_NaN = df[row_has_NaN]

rows_with_NaN
df.dropna(inplace=True)

df.isnull().sum()
df.info()
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
sns.pairplot(df)
df.Season.value_counts().plot(kind="bar")
player_of_match = df["player_of_match"].value_counts()[:10]

player_of_match.plot(kind="barh")

print(df["player_of_match"].value_counts()[:10])
match_winner = df["winner"].value_counts()

match_winner.plot(kind="barh")

print(df["winner"].value_counts())
df.loc[df.Season == 2019, "winner"].value_counts().plot(kind="barh")
df["toss_decision"].value_counts().plot(kind="barh")

df["toss_decision"].value_counts()
df.loc[df.win_by_runs > 50, "winner"].value_counts().plot(kind="barh")
df.loc[df.win_by_wickets > 5, "winner"].value_counts().plot(kind="barh")
df.loc[df.toss_winner == df.winner, "winner"].value_counts().plot(kind="barh")

print(df.loc[df.toss_winner == df.winner, "winner"].value_counts())
teams = df.team1.unique().tolist()

teams.sort()

for team1 in teams:

    for team2 in df.team2.unique().tolist():

        df_ttw = df.loc[(df["team1"] == team1) & (df["team2"] == team2), "winner"]

        if len(df_ttw) > 0:

            print(df_ttw.value_counts())
df.loc[(df["team1"] == "Chennai Super Kings") & (df["team2"] == "Mumbai Indians"), "winner"].value_counts().plot(kind="pie")
df = df[df['city'].notna()]

city = df.city.unique().tolist()

city.remove("-")
#ec38974f44884f42bfa871dc36b8a090

!pip install opencage

from opencage.geocoder import OpenCageGeocode

import folium

key = "ec38974f44884f42bfa871dc36b8a090"  

geocoder = OpenCageGeocode(key)

india = geocoder.geocode("India")

lat = india[0]['geometry']['lat']

lng = india[0]['geometry']['lng']

map = folium.Map(location=[lat, lng], zoom_start=2)

for query in city:

    pop = query

    if query == "Kochi":

        query = "Kochi India"

    results = geocoder.geocode(query)

    lat = results[0]['geometry']['lat']

    lng = results[0]['geometry']['lng']

    folium.Marker((lat, lng), popup=pop).add_to(map)

map