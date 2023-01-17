import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
PATH = '../input/zomato-bangalore-restaurants/'

df = pd.read_csv(f"{PATH}/zomato.csv")
df.head(5)
#checking all the columns of the data

df.columns
#finding all the restaurants in Koramangala 

df_koramangala = df[df['location'] ==  \

                    'Koramangala']
df_koramangala.columns
popular_df = df_koramangala[['name','rate', 'approx_cost(for two people)', 'online_order', 'votes', 'cuisines']]
popular_df['rating'] = popular_df.rate.astype('str')

popular_df['rating'] = popular_df['rating'].apply(lambda x : float(x.split('/')[0]) if x!= '-' else 0.0)

popular_df = popular_df.fillna(0)
popular_df.head(5)
#Number of outlets

pop_counts = popular_df.name.value_counts()

pop_counts
sns.barplot(x = pop_counts, y = pop_counts.index)
#checking the most popular cuisines in Koramangala

plt.figure(figsize = (8,8))

cuisines = df_koramangala['cuisines'].value_counts()[:10]

sns.barplot(cuisines, cuisines.index)

plt.xlabel('Count')

plt.ylabel('Cuisine')
df_koramangala[df['name'] == 'Hunger Hitman'].iloc[1][0]