import numpy as np

import pandas as pd

import os

import json

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

%matplotlib inline

inline_rc = dict(mpl.rcParams)


import os

print(os.listdir("../input"))
# We only use the first 100,000 data in this assignment

users = []

with open('../input/yelp-dataset/yelp_academic_dataset_business.json') as fl:

    for i, line in enumerate(fl):

        users.append(json.loads(line))

        if i+1 >= 100000:

            break

df = pd.DataFrame(users)

df.head()
x=df['stars'].value_counts()

x=x.sort_index()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Star Rating Distribution")

plt.ylabel('# of businesses', fontsize=12)

plt.xlabel('Star Ratings ', fontsize=12)
business_cats = ''.join(df['categories'].astype('str'))



cats=pd.DataFrame(business_cats.split(','),columns=['categories'])



#prep for chart

x=cats.categories.value_counts()



x=x.sort_values(ascending=False)

x=x.iloc[0:20]



#chart

plt.figure(figsize=(16,4))

ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])

plt.title("What are the top categories?",fontsize=25)

locs, labels = plt.xticks()

plt.setp(labels, rotation=80)

plt.ylabel('# businesses', fontsize=12)

plt.xlabel('Category', fontsize=12)



#adding the text labels

# rects = ax.patches

# labels = x.values

# for rect, label in zip(rects, labels):

#     height = rect.get_height()

#     ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()



# Measures of central tendency for given data

df.describe()
#Get the distribution of the ratings

x=df['city'].value_counts()

x=x.sort_values(ascending=False)

x=x.iloc[0:20]

plt.figure(figsize=(16,4))

ax = sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Which city has the most reviews?")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.ylabel('# businesses', fontsize=12)

plt.xlabel('City', fontsize=12)
#Get the distribution of the ratings

x=df['city'].value_counts()

x=x.sort_values(ascending=False)

x=x.iloc[0:20]

plt.figure(figsize=(16,4))

ax = sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Which city has the most reviews?")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.ylabel('# businesses', fontsize=12)

plt.xlabel('City', fontsize=12)
sns.kdeplot(df.sample(10000).review_count,df.sample(10000).stars,shade=True)
sns.distplot(df.review_count,kde=True,hist=True)
sns.distplot(df.stars,kde=True,hist=True)