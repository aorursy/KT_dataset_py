# imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


%matplotlib inline

# Read the data
medium_articles = pd.read_csv("../input/medium-articles-dataset/medium_data.csv")
print(medium_articles.head())
medium_articles.info()
medium_articles.describe()
# Publications and total articles count

total_articles = medium_articles['publication'].value_counts()
total_articles = dict(total_articles)

publication_name = [p for p in total_articles]
article_count = [c for c in total_articles.values()]

# plot the total articles count data

fig, ax = plt.subplots()
ax = sns.set_style('darkgrid')
ax = sns.barplot(x=publication_name, y=article_count)
ax.set_xticklabels(publication_name, rotation=90)

plt.title("Publications & total number of articles", fontsize=16)
plt.xlabel('Publications', fontsize=12)
plt.ylabel('Article count', fontsize=12)
plt.show()
# Count the total claps for each publication
total_claps = medium_articles.groupby('publication')['claps'].sum()
total_claps = dict(total_claps)

publication_name = [p for p in total_claps]
claps_count = [c for c in total_claps.values()]

# Plot the total claps data
fig, ax = plt.subplots()
ax = sns.set_style('darkgrid')
ax = sns.barplot(x=publication_name, y=claps_count)
ax.set_xticklabels(publication_name, rotation=90)

plt.title("Publications & total claps", fontsize=16)
plt.xlabel('Publications', fontsize=12)
plt.ylabel('Total Claps', fontsize=12)
plt.show()
# Top 10 articles
top_20_articles = medium_articles.sort_values(by='claps', ascending=False)
top_20_articles = top_20_articles[['title', 'claps', 'responses', 'publication', 'url']]
pd.set_option('display.max_colwidth', 1)

top_20_articles.head(20)
# Average claps for the articles without an image
without_image = medium_articles[medium_articles['image'].isnull()]
without_image_claps_avg = round(without_image['claps'].mean())
print(without_image_claps_avg)

# Average claps for the articles with an image
with_image = medium_articles[medium_articles['image'].notnull()]
with_image_claps_avg = round(with_image['claps'].mean())
print(with_image_claps_avg)


x_axis = ['Yes', 'No']
y_axis = [with_image_claps_avg, without_image_claps_avg]


# plot the date(with & without image)
fig, ax = plt.subplots()
ax = sns.barplot(x=x_axis, y=y_axis)

plt.title('Claps & Wheather the article has an image', fontsize=16)
plt.xlabel('Image', fontsize=12)
plt.ylabel('Average Claps', fontsize=12)
plt.show()
# In the response column the value 'Read' cause an error when changing the dtype to int64
# Therefore replace 'Read' with '0' then change the dtype to int64
responses_clean = medium_articles['responses']
responses_clean = responses_clean.replace({'Read': '0'}).astype('int64')

claps = medium_articles['claps']

# Claps and responses distribution
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.boxplot(claps)
ax2.boxplot(responses_clean)

ax1.set_title('Claps Distribution')
ax2.set_title('Responses Distribution')

plt.show()
# Create a new column for the published month
medium_articles['published_month'] = pd.DatetimeIndex(medium_articles['date']).month
sorted_by_pub_month = medium_articles.groupby(
    ['publication', 'published_month'])['claps'].sum().reset_index(name='monthly_total_claps')
sorted_by_pub_month
# plot the total claps throughout the year
publications = list(sorted_by_pub_month['publication'].unique())

fig, ax = plt.subplots(figsize=(16, 10))

ax.set_title('Claps throughout the year', fontsize=16)

for pub in publications:
    publication_monthly = sorted_by_pub_month['publication'] == pub
    publication_monthly_data = sorted_by_pub_month.loc[publication_monthly, 
                                                       ['published_month', 'monthly_total_claps']]
    ax = sns.lineplot(publication_monthly_data['published_month'], 
                      publication_monthly_data['monthly_total_claps'], label=pub, linewidth=3)
    
    ax.set_xlabel('Months',fontsize=14)
    ax.set_ylabel('Total Claps', fontsize=14)
    ax.legend()
    
