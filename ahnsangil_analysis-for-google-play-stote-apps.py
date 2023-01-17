import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import matplotlib.pyplot as plt
df_main = pd.read_csv('../input/googleplaystore.csv')

df_user = pd.read_csv('../input/googleplaystore_user_reviews.csv')
df_main.head()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.set_title('Top 10 App Categories in Google Play Store', fontsize=15, weight='heavy')

df_main.Category.value_counts()[:10].sort_values(ascending=True).plot(kind='barh')

plt.show()
df_main.Installs.unique()
df_main.loc[:, 'Installs'] = df_main['Installs'].str.replace('+', '')

df_main.loc[:, 'Installs'] = df_main['Installs'].str.replace(',', '')

df_main.loc[:, 'Installs'] = df_main['Installs'].str.replace('Free', '0')

df_main.loc[:, 'Installs'] = df_main['Installs'].map(int)
df_main.Installs.unique()
df_main.Installs.mean()
df_main_50m = df_main[df_main.Installs >= df_main.Installs.mean()]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

df_main_50m.Category.value_counts()[:10].sort_values(ascending=True).plot(kind='barh')

ax.set_title('Top 10 App Categories over 50M installs in Google Play Store', fontsize=15, weight='heavy')

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

df_main_50m.groupby('Category')['Rating'].mean().sort_values(ascending=False)[:10].sort_values(ascending=True).plot(kind='barh')

ax.set_title('Top 10 App Categories over 50M installs in Google Play Store', fontsize=15, weight='heavy')

plt.show()
df_main.drop_duplicates(['App']).sort_values(by='Reviews', ascending=False)[:20]
df_main.loc[:, 'Reviews'] = df_main.Reviews.map(int)

df_main.drop_duplicates(['App']).sort_values(by='Reviews', ascending=False)[:20]
df_main['Reviews'].isin(['3.0M']).sum()
(df_main.Reviews.str.isdigit() == False).sum()
df_main.loc[:, 'Reviews'] = df_main['Reviews'].str.replace('3.0M', '3000000')

df_main.loc[:, 'Reviews'] = df_main.Reviews.map(int)
df_main.drop_duplicates(['App']).sort_values(by='Reviews', ascending=False)[:20]
df_main.Price.unique()
df_main.loc[:, 'Price'] = df_main.Price.str.replace('Everyone', '0')

df_main.loc[:, 'Price'] = df_main.Price.str.replace('$', '')

df_main.loc[:, 'Price'] = df_main.Price.map(float)
df_main.Price.unique()
df_main['Sales'] = df_main['Installs'] * df_main['Price']
def get_sales_prop(group):

    group = group.sort_values(by='Sales', ascending=False)

    group['Sales_prop'] = group['Sales'] / group['Sales'].sum()

    group['Sales_prop_cum'] = group['Sales_prop'].cumsum()

    return group
df_main = df_main.drop_duplicates('App')
df_main = df_main.groupby('Category').apply(get_sales_prop)
df_top10_sales = df_main.sort_values(by='Sales', ascending=False)[:10]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

df_top10_sales.set_index('App')['Sales'].sort_values(ascending=True).plot(kind='barh', ax=ax)

ax.set_title('Top 10 Sales Apps in Google Play Store', fontsize=20, weight='heavy')

ax.set_xlabel('Sales($10,000,000)', fontsize=15, color='red')

ax.set_ylabel('Apps', fontsize=15, color='green', rotation=0)

plt.show()
df_user.head()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

df_user.Sentiment.value_counts(ascending=True).plot(kind='barh', ax=ax)

ax.set_title('App Users Sentiment Analysis in Google Play store', fontsize=20, weight='heavy')

ax.set_xlabel('Sentiment numbers', fontsize=15)

ax.set_ylabel('Sentiments', fontsize=15)

plt.show()
df_user['Sentiment_value'] = np.where(df_user['Sentiment'] == 'Positive', 1,

                                      np.where(df_user['Sentiment'] == 'Negative', -1, 0))
df_user.head()
df_user_best15 = df_user.groupby('App')['Sentiment_value'].sum().sort_values(ascending=False)[:15].sort_values(ascending=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

df_user_best15.plot(kind='barh', ax=ax)

ax.set_title('Best 15 Apps by Users Sentiment in Google Play store', fontsize=20, weight='heavy')

ax.set_xlabel('Sentiment values', fontsize=15)

ax.set_ylabel('Apps', fontsize=15, rotation=0)

plt.show()
df_user_best15 = df_user.groupby('App')['Sentiment_value'].sum().sort_values()[:15].sort_values(ascending=False)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

df_user_best15.plot(kind='barh', ax=ax)

ax.set_title('Worst 15 Apps by Users Sentiment in Google Play store', fontsize=20, weight='heavy')

ax.set_xlabel('Sentiment values', fontsize=15)

ax.set_ylabel('Apps', fontsize=15, rotation=0)

plt.show()
df_game = df_main[df_main['Category'] == 'GAME']
df_game.head()
sentiment_dict = df_user.groupby('App')['Sentiment_value'].sum().to_dict()
df_game['Sentiment_value'] = df_game['App'].map(sentiment_dict)
df_game.head()
df_game_best15 = df_game.groupby('App')['Sentiment_value'].sum().sort_values(ascending=False)[:15].sort_values(ascending=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

df_game_best15.plot(kind='barh', ax=ax)

ax.set_title('Best 15 GAME Apps by Users Sentiment in Google Play store', fontsize=20, weight='heavy')

ax.set_xlabel('Sentiment values', fontsize=15)

ax.set_ylabel('GAME Apps', fontsize=15, rotation=0)

plt.show()
df_game_worst15 = df_game.groupby('App')['Sentiment_value'].sum().sort_values()[:15].sort_values(ascending=False)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

df_game_worst15.plot(kind='barh', ax=ax)

ax.set_title('Worst 15 GAME Apps by Users Sentiment in Google Play store', fontsize=20, weight='heavy')

ax.set_xlabel('Sentiment values', fontsize=15)

ax.set_ylabel('GAME Apps', fontsize=15, rotation=0)

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

df_game_genre_install = df_game.groupby('Genres')['Installs'].sum().sort_values(ascending=False)[:10].sort_values(ascending=True)

df_game_genre_install.plot(kind='barh', ax=axes[0])

axes[0].set_title('Best 10 GAME Genres by Installs', fontsize=15, weight='heavy')

axes[0].set_xlabel('Installs', fontsize=15)

axes[0].set_ylabel('Genres', fontsize=15)

df_game_genre_sales = df_game.groupby('Genres')['Sales'].sum().sort_values(ascending=False)[:10].sort_values(ascending=True)

df_game_genre_sales.plot(kind='barh', ax=axes[1])

axes[1].set_title('Best 10 GAME Genres by Sales', fontsize=15, weight='heavy')

axes[1].set_xlabel('Sales($10,000,000)', fontsize=15)

plt.subplots_adjust(top=0.8, bottom=0.2, wspace=0.4)

plt.show()
df_game['Sales_prop'] = df_game['Sales'] / df_game['Sales'].sum()
df_game['Sales_prop_cum'] = df_game['Sales_prop'].cumsum()
df_game.head(3)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sales_prop = df_game.set_index(['App'])['Sales_prop'][:10]

sales_prop_cum = df_game.set_index(['App'])['Sales_prop_cum'][:10]

sales_prop.plot(kind='bar', ax=ax, alpha=0.5, rot=90)

sales_prop_cum.plot(kind='line', ax=ax, style='ko--', rot=90)

ax.set_title("Best 10 GAME Apps' Sales Proportion", fontsize=15, weight='heavy')

ax.set_xlabel('GAME App', fontsize=15)

ax.set_ylabel('Sales proportion', fontsize=15)

ax.legend()

plt.show()
df_main = df_main.reset_index(drop=True)
for category in df_main.Category.unique():

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    df = df_main[df_main['Category'] == category]

    sales_prop = df.set_index(['App'])['Sales_prop'][:10]

    sales_prop_cum = df.set_index(['App'])['Sales_prop_cum'][:10]

    sales_prop.plot(kind='bar', ax=ax, alpha=0.5, rot=90)

    sales_prop_cum.plot(kind='line', ax=ax, style='ko--', rot=90)

    ax.set_title("Best 10 " + category + " Apps' Sales Proportion", fontsize=15, weight='heavy')

    ax.set_xlabel(category + ' App', fontsize=15)

    ax.set_ylabel('Sales proportion', fontsize=15)

    ax.legend()

plt.show()