import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt; plt.rcdefaults()
Seasons_Stats = pd.read_csv("../input/Seasons_Stats cleaned.csv")
Seasons_Stats.columns
Seasons_Stats = Seasons_Stats [['Year', 'Player',  'Age', 'G',  'MP']]



# G = Game played per season

# MP = Minutes played per season
Seasons_Stats.head()
Seasons_sum = Seasons_Stats.groupby('Year', as_index=False)["G","MP"].sum()

Seasons_sum.head()                             
Seasons_yearly = pd.DataFrame({'Year': Seasons_sum['Year'], 'Avg': Seasons_sum['MP']/Seasons_sum['G']})

Seasons_yearly.head()
Seasons_yearly.plot(kind='line',x='Year',y='Avg', color='red', title='Average MP for each year')
Seasons_age = Seasons_Stats.groupby('Year', as_index=False).agg({'Age': 'min'})

Seasons_age.head()
Seasons_age.plot(kind='line',x='Year',y='Age', color='red',title='Min age for each year')
Seasons_age = Seasons_Stats.groupby('Year', as_index=False).agg({'Age': 'max'})

Seasons_age.head()
Seasons_age.plot(kind='line',x='Year',y='Age', color='red',title='Max age for each year')
Seasons_age = Seasons_Stats.groupby('Year', as_index=False).agg({'Age': 'mean'})

Seasons_age.head()
Seasons_age.plot(kind='line',x='Year',y='Age', color='red',title='Average age for each year')