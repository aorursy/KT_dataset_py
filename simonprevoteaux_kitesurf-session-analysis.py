from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import re

import seaborn as sns
# read the file and parse the first and only sheet (need python xlrd module)

f = pd.ExcelFile('../input/foil_data_raw.xlsx')

data = f.parse(parse_cols=[i for i in range(0,15)], names=["Date","Duree","Lieu","Vent","Orientation","Twin Tip","Surf","Foil","15m","12m","11m","10m","7m","5m","Commentaire"])

data = data.fillna(0)

data.head()
data.describe()
nb_spots = 10

spots = data.groupby('Lieu')['Lieu'].agg('count').sort_values(ascending=False)

spots = pd.DataFrame({'Spot': spots.index, 'time rided': spots.values})[0:nb_spots]





sns.barplot(y = 'Spot', x = 'time rided', data = spots )

top = 5

spots = data.groupby('Lieu')['Lieu'].agg('count').sort_values(ascending=False)[0:top]

mask = data['Lieu'].isin(spots.index)

spots = data.loc[mask]



sns.countplot(x="Lieu", hue="Orientation", data=spots)
