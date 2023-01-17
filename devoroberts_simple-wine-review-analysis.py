import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
reviews = pd.read_csv('../input/winemag-data-130k-v2.csv')
reviews.head()
reviews.describe().T
plt.subplots(figsize=(20,10))
sns.countplot('points', data=reviews)
sns.set(font_scale=2)
plt.title('Points Count')
plt.xlabel('Points')
plt.ylabel("Count")
sns.kdeplot(reviews['points'])
plt.title('Points Count')
plt.xlabel('Points')
plt.ylabel("Count")
sns.lmplot('points','price', data=reviews)
rev_count_df = pd.DataFrame(reviews.groupby(['country'])['taster_name'].count())
rev_count_df.head(10)
rev_count_df.describe()
rev_count_df.sort_values(by='taster_name', ascending=False, inplace=True)
rev_count_df.head(10)
top_reviewed = rev_count_df.iloc[:10,:]
top_reviewed
plt.subplots(figsize=(20,10))
sns.set(font_scale=2)
r_count = sns.barplot(top_reviewed['taster_name'],top_reviewed.index, data=top_reviewed, palette='winter')
r_count.set_title('Review Count by Country (Top 10)', fontsize=30)
r_count.set_xlabel('Review Count', fontsize=30)
r_count.set_ylabel('Country', fontsize=30)

reviews.head()
prices_df = reviews[['country', 'price']]
prices_before = prices_df['price'].count()
prices_df = prices_df.dropna()
prices_after = prices_df['price'].count()
prices_before - prices_after
prices_df.head()
avg_price = pd.DataFrame(prices_df.groupby('country')['price'].mean())
avg_price.sort_values(by='price', ascending=False, inplace=True)
avg_price.head(10)
top_avg = avg_price.iloc[:10,:]
plt.subplots(figsize=(20,10))
sns.set(font_scale=2)
wine_avg = sns.barplot(top_avg['price'],top_avg.index, data=top_avg, palette='summer')
wine_avg.set_title('Avg Price by Country (Top 10)', fontsize=30)
wine_avg.set_xlabel('Avg Price', fontsize=30)
wine_avg.set_ylabel('Country', fontsize=30)
reviews.head()
wine_variety = reviews[['country', 'variety']]
wine_variety = wine_variety.dropna()
wine_variety.head()
variety_count = pd.DataFrame(wine_variety.groupby('variety')['country'].count())
variety_count.head()
variety_count.sort_values(by='country', ascending=False, inplace=True)
#sns.barplot(variety_count['country'], variety_count.index, data=variety_count, palette='GnBu_d')
top_variety = variety_count.iloc[:10,:]
top_variety.columns = ['Count']
top_variety.head(10)
plt.subplots(figsize=(20,10))
var_count = sns.barplot(top_variety.Count, top_variety.index, palette='cubehelix')
sns.set(font_scale=2)
var_count.set_title('Popular Wine Varieties (Top 10)', fontsize=30)
var_count.set_xlabel('Variety Count', fontsize=30)
var_count.set_ylabel('Varieties', fontsize=30)
