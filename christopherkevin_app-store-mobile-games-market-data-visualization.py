# Importing libraries

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import math
df_raw = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

df_raw.head()
print(f"Rows: {df_raw.shape[0]} \nColumns: {df_raw.shape[1]}")
df = df_raw.loc[:, ["Name","Average User Rating", "User Rating Count", "Price", "In-app Purchases"]]

df.head()
df = df.loc[(df['User Rating Count'] >= 500)]

print(f"Rows: {df.shape[0]} \nColumns: {df.shape[1]}")

df.head()
df['Monetization'] = 'Free'

df.loc[(df['Price'] == 0) & (pd.notnull(df['In-app Purchases'])), 'Monetization'] = 'Free, In-App Purchases'

df.loc[(df['Price'] > 0) & (pd.isnull(df['In-app Purchases'])), 'Monetization'] = 'Paid'

df.loc[(df['Price'] > 0) & (pd.notnull(df['In-app Purchases'])), 'Monetization'] = 'Paid, In-App Purchases'



df.head()
sns.catplot(data=df, x='Average User Rating', y='User Rating Count', kind= 'bar', hue='Monetization', legend_out=True, ci=None, palette='plasma', aspect=2.5)

plt.title("User Rating Count per Average User Rating Comparison between Monetization Methods")



plt.show()
free_df = df.loc[df['Monetization'] == 'Free'].groupby(by=df['Average User Rating']).sum()

freeIAP_df = df.loc[df['Monetization'] == 'Free, In-App Purchases'].groupby(by=df['Average User Rating']).sum()

paid_df = df.loc[df['Monetization'] == 'Paid'].groupby(by=df['Average User Rating']).sum()

paidIAP_df = df.loc[df['Monetization'] == 'Paid, In-App Purchases'].groupby(by=df['Average User Rating']).sum()
fig, axs = plt.subplots(2, 2, figsize=(15,15))



themeA = plt.get_cmap('PuRd')

themeB = plt.get_cmap('Oranges')

themeC = plt.get_cmap('BuPu')

themeD = plt.get_cmap('YlGn')



axs[0,0].set_prop_cycle("color", [themeA(1. * i / len(free_df))

                              for i in range(len(free_df))])

axs[0,1].set_prop_cycle("color", [themeB(1. * i / len(freeIAP_df))

                              for i in range(len(freeIAP_df))])

axs[1,0].set_prop_cycle("color", [themeC(1. * i / len(paid_df))

                              for i in range(len(paid_df))])

axs[1,1].set_prop_cycle("color", [themeD(1. * i / len(paidIAP_df))

                              for i in range(len(paidIAP_df))])



axs[0,0].pie(free_df['User Rating Count'], labels=free_df.index, autopct='%1.1f%%') 

axs[0,0].set_title('Free Games User Rating Composition')



axs[0,1].pie(freeIAP_df['User Rating Count'], labels=freeIAP_df.index, autopct='%1.1f%%') 

axs[0,1].set_title('In-App Purchased Games User Rating Composition')



axs[1,0].pie(paid_df['User Rating Count'], labels=paid_df.index, autopct='%1.1f%%') 

axs[1,0].set_title('Paid Games User Rating Composition')



axs[1,1].pie(paidIAP_df['User Rating Count'], labels=paidIAP_df.index, autopct='%1.1f%%') 

axs[1,1].set_title('Paid + In-App Purchased Games User Rating Composition')



plt.show()
df = df_raw.loc[(df_raw['User Rating Count'] >= 200) & (df_raw['Primary Genre']=='Games')]

df = df.loc[:, ["Name","Price", "Size", "Original Release Date"]]

df.head()
df['Release Year'] = pd.to_datetime(df["Original Release Date"]).dt.year

df['Size'] = round(df['Size']/1000000)

df.head()
groupedByYear_df = df.groupby(by=df['Release Year']).count()

groupedByYear_df.head(10)



fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(111)

ax1 = sns.lineplot(x=groupedByYear_df.index, y='Price', data=groupedByYear_df, color='Orange')

ax1.set_ylabel('Count')



plt.title('Release Year vs. Price and Size Comparison')

plt.show()
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(111)

ax1 = sns.lineplot(x='Release Year', y='Price', data=df, color='Orange')



ax2 = ax1.twinx()

ax2 = sns.lineplot(x='Release Year', y='Size', data=df, color='Blue')



from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='Orange', lw=4),

                Line2D([0], [0], color='Blue', lw=4)]

ax2.legend(custom_lines, ['Release Year vs. Price', 'Release Year vs. Size'])



plt.title('Release Year vs. Price and Size Comparison')

plt.show()