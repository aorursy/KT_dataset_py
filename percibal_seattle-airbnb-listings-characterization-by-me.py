# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np                # algebra lineal

import pandas as pd               # data frames

import seaborn as sns             # gráficos

import matplotlib.pyplot as plt   # gráficos

import scipy.stats                # estadísticas

from datetime import datetime

from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
listings = pd.read_csv("../input/dataset/listings2.csv")

# We don't use this information for this first exercise

# prices = pd.read_csv("../input/calendar.csv")

# reviews = pd.read_csv("../input/reviews.csv")



listings.head()
lDf = listings.loc[:, ["id", "review_scores_location", "price", "bedrooms", "first_review", "neighbourhood_cleansed"]]



print("_ COLUMNS INFO _")

print(lDf.info())



lDf.head()
# Scrores Rating change 0 for the mean and divide by 100

lDf['review_scores_location'] = lDf['review_scores_location'].fillna(-1)

lDf['review_scores_location'].replace('', -1, inplace=True)

lDf['first_review'] = lDf['first_review'].fillna(0)



# Format currency to number

lDf['price'] = lDf['price'].replace('[\$,]', '', regex=True).astype(float)

# Change 0 bedrooms to 1

lDf['bedrooms'] = lDf['bedrooms'].replace(to_replace=0,value=1)

# Normalize price according to bedrooms

lDf["price_norm"] = lDf["price"] / lDf["bedrooms"] 



def isNaN(num):

    return num != num



# Reduce columns

lDf = lDf.loc[:, ["id", "review_scores_location", "price_norm", "first_review", "neighbourhood_cleansed"]]

for i in lDf.index:

    date_review = lDf.at[i,'first_review']

    score = lDf.at[i,'review_scores_location']

    nGhood = lDf.at[i,'neighbourhood_cleansed'] 

    px = lDf.at[i,'price_norm'] 

    # Format date in elapsed months

    if(date_review!=0):

        date_review = datetime.strptime(date_review, '%Y-%M-%d')

        date_review = (datetime.now() - date_review).days

        lDf.at[i, 'first_review'] = float(date_review/30)

    # Replace not reviewed listings location score by its neighbourhood average

    if(score==-1):

        score_mean = lDf[ (lDf.neighbourhood_cleansed == nGhood) 

                         & (lDf.review_scores_location > 0) ].mean()['review_scores_location']

        lDf.at[i,'review_scores_location'] = score_mean

    if(isNaN(px)):

         lDf.at[i,'price_norm'] = lDf[ (lDf.neighbourhood_cleansed == nGhood) ].mean()['price_norm']

    

    

# Print transformed DataSet normalized and without outliers

lDf.head()
from scipy.cluster.hierarchy import fcluster

# Neighbourhoods Data Frame

nLDf = lDf.loc[:, ["neighbourhood_cleansed", "review_scores_location"]]



meanSFD = nLDf.groupby(['neighbourhood_cleansed'], as_index=False).mean()

meanSFD.head()

# Neighbourhoods ranking

neigsDs = meanSFD.sort_values(['review_scores_location'],ascending=False).reset_index(drop=False)

plt.figure(figsize=(20,20))

sns.barplot(x=neigsDs["review_scores_location"],y=neigsDs["neighbourhood_cleansed"])

plt.xlabel("Scores",fontsize=15)

plt.ylabel("Neighbourhoods",fontsize=15)

plt.title("Best Locations in Seattle - Airbnb.com",fontsize=15)

plt.show()
with warnings.catch_warnings():

    # Compare location score vs. price

    sns.lmplot(x="review_scores_location",y="price_norm",data=lDf)
dist_sin = linkage(lDf.loc[:,["price_norm","review_scores_location","first_review"]],method="single")

plt.figure(figsize=(18,6))

dendrogram(dist_sin, leaf_rotation=90)

plt.xlabel('Index')

plt.ylabel('D')

plt.suptitle("DENDROGRAM SINGLE METHOD",fontsize=18)

plt.show()
dist_comp = linkage(lDf.loc[:,["price_norm","review_scores_location","first_review"]],method="ward")



plt.figure(figsize=(25,10))

dendrogram(dist_comp, leaf_rotation=120)

plt.xlabel('sample index')

plt.ylabel('distance')

plt.suptitle("Hierarchical Clustering Dendrogram - Complete Method",fontsize=20) 

plt.show()
lDf_D=lDf.copy()

lDf_D['2_clust']=fcluster(dist_comp,2, criterion='maxclust')

lDf_D['3_clust']=fcluster(dist_comp,3, criterion='maxclust')

lDf_D['4_clust']=fcluster(dist_comp,4, criterion='maxclust')

lDf_D.head()



plt.figure(figsize=(24,7))



plt.suptitle("Hierarchical Clustering CM",fontsize=25)



plt.subplot(1,4,1)

plt.title("K = 2",fontsize=20)

sns.scatterplot(x="price_norm",y="review_scores_location", data=lDf_D, hue="2_clust",palette="Paired")



plt.subplot(1,4,2)

plt.title("K = 3",fontsize=20)

sns.scatterplot(x="price_norm",y="review_scores_location", data=lDf_D, hue="3_clust",palette="Paired")



plt.subplot(1,4,3)

plt.title("K = 4",fontsize=20)

sns.scatterplot(x="price_norm",y="review_scores_location", data=lDf_D, hue="4_clust",palette="Paired")