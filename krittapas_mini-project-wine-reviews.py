import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

sns.set_context('notebook', font_scale=1.5)

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

print("Before removing duplicates:", len(wine_reviews))

wine_reviews.tail()
wine_reviews = wine_reviews.drop_duplicates()

print("Removing duplicates based on all columns:", len(wine_reviews))
wine_reviews_ddp = wine_reviews.drop_duplicates('description')

print("Removing duplicates based on description:", len(wine_reviews_ddp))
wine_reviews_all = wine_reviews.merge(wine_reviews_ddp, how='outer', indicator=True)

dup_wine_desc = wine_reviews_all[wine_reviews_all['_merge']=='left_only'].description



wine_reviews_all[wine_reviews_all['description'].isin(dup_wine_desc)]
wine_reviews.describe()
print('Skewness=%.3f' %wine_reviews['points'].skew())

print('Kurtosis=%.3f' %wine_reviews['points'].kurtosis())

sns.distplot(wine_reviews['points'], bins=20, kde=True);
print('Skewness=%.3f' %wine_reviews['price'].skew())

print('Kurtosis=%.3f' %wine_reviews['price'].kurtosis())

sns.distplot(wine_reviews['price'].dropna());
print('Skewness=%.3f' %np.log(wine_reviews['price']).skew())

print('Kurtosis=%.3f' %np.log(wine_reviews['price']).kurtosis())

sns.distplot(np.log(wine_reviews['price']).dropna());
sns.set(style = 'whitegrid', rc = {'figure.figsize':(8,6), 'axes.labelsize':12})

sns.scatterplot(x = 'price', y = 'points', data = wine_reviews);
sns.boxplot(x = 'points', y = 'price', palette = 'Set2', data = wine_reviews, linewidth = 1.5);
wine_cat = wine_reviews.select_dtypes(include=['object']).columns

print('n rows: %s' %len(wine_reviews))

for i in range(len(wine_cat)):

    c = wine_cat[i]

    print(c, ': %s' %len(wine_reviews[c].unique()))
wine_reviews['country'].value_counts()
print(wine_reviews['region_2'].isna().sum())

wine_reviews['region_2'].value_counts()