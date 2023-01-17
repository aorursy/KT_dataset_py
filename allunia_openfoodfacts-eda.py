import numpy as np 
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/en.openfoodfacts.org.products.tsv", sep="\t", nrows=100000)
data.shape
data.head()
percent_of_nans = data.isnull().sum().sort_values(ascending=False) / data.shape[0] * 100
plt.figure(figsize=(10,5))
sns.distplot(percent_of_nans, bins=100, kde=False)
plt.xlabel("% nans")
plt.ylabel("Number of columns")
plt.title("Percentage of nans per feature column")
useless_features = percent_of_nans[percent_of_nans == 100].index
useless_features
len(useless_features)
data.drop(useless_features, axis=1, inplace=True)
data.shape
zero_nan_features = percent_of_nans[percent_of_nans == 0].index
zero_nan_features
example = data.loc[0,zero_nan_features]
print(example["states_tags"])
print(example["states_en"])
example.loc[['created_datetime', 'last_modified_datetime']]
example.loc["last_modified_t"] - example.loc["created_t"]
low_nans = percent_of_nans[percent_of_nans <= 15]
middle_nans = percent_of_nans[(percent_of_nans > 15) & (percent_of_nans <= 50)]
high_nans = percent_of_nans[(percent_of_nans > 50) & (percent_of_nans < 100)]
def rotate_labels(axes):
    for item in axes.get_xticklabels():
        item.set_rotation(45)
plt.figure(figsize=(20,5))
lows = sns.barplot(x=low_nans.index.values, y=low_nans.values, palette="Greens")
rotate_labels(lows)
plt.title("Features with fewest nan-values")
plt.ylabel("% of nans ")
data.loc[1,'additives_n']
data.loc[1,'additives']
data.loc[1,'ingredients_text']
data.loc[1,['countries', 'countries_tags', 'countries_en']]
plt.figure(figsize=(20,5))
middle = sns.barplot(x=middle_nans.index.values, y=middle_nans.values, palette="Oranges")
rotate_labels(middle)
plt.title("Features with medium number of nan-values")
plt.ylabel("% of nans ")
data.loc[7,['additives', 'additives_n', 'additives_tags']]
data.loc[7,['ingredients_text']].values
plt.figure(figsize=(15,30))
high = sns.barplot(y=high_nans.index.values, x=high_nans.values, palette="Reds")
plt.title("Features with most nan-values")
plt.ylabel("% of nans ")
data[data['allergens'].isnull() == False][['allergens','ingredients_text']][0:10]
data[data['categories'].isnull() == False].categories
data.drop(high_nans.index, axis=1, inplace=True)
data.shape
data.to_csv("cropped_open_food_facts.csv")