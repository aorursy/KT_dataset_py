# Data wrangling and math

import numpy as np

import pandas as pd 



# Visualization tools

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

%matplotlib inline
df_ayam = pd.read_csv('../input/indonesian-food-recipes/dataset-ayam.csv')

df_ikan = pd.read_csv('../input/indonesian-food-recipes/dataset-ikan.csv')

df_kambing = pd.read_csv('../input/indonesian-food-recipes/dataset-kambing.csv')

df_sapi = pd.read_csv('../input/indonesian-food-recipes/dataset-sapi.csv')

df_tahu = pd.read_csv('../input/indonesian-food-recipes/dataset-tahu.csv')

df_telur = pd.read_csv('../input/indonesian-food-recipes/dataset-telur.csv')

df_tempe = pd.read_csv('../input/indonesian-food-recipes/dataset-tempe.csv')

df_udang = pd.read_csv('../input/indonesian-food-recipes/dataset-udang.csv')



df = df_ayam.append([df_ikan,df_kambing,df_sapi,df_tahu,df_telur,df_tempe,df_udang])
df
print(f'There are {df.shape[0]} recipes (rows) and {df.shape[1]} columns.')

print(f'Columns: {df.columns.values}')
df.drop('URL', axis=1, inplace=True)
loved_food = df.sort_values(by='Loves', ascending=False)[:10]

figure = plt.figure(figsize=(10,6))

sns.barplot(y=loved_food.Title, x=loved_food.Loves)

plt.xticks()

plt.xlabel('Food recipe')

plt.ylabel('Loves count')

plt.title('The Most Loved Food Recipe')

plt.show()