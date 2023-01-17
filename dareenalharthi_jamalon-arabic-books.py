

import re

import requests

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Create DataFrame 

books_df = pd.read_csv('../input/jamalon-arabic-books-dataset/jamalon dataset.csv')
books_df.head()
# Replace the empty cells with nan value to detect the null cells

books_df.replace('^\s+$', np.nan, inplace=True)

books_df.isnull().sum()
books_df.info()
books_df.shape
# Check columns types

books_df.dtypes
# Count the books in each category

indices = books_df.Category.value_counts().index

count = books_df.Category.value_counts().values

categories = []

# Reshape categories Arabic names in readable shape  

plt.figure(figsize=(18,8))

sns.barplot(x = indices, y=count)

plt.title("\nThe number of books in each category\n", fontdict={'fontsize':20})

plt.xlabel('Categories',fontdict={'fontsize':20})

plt.ylabel('Number of books',fontdict={'fontsize':20})

plt.show()
x = books_df[~(books_df['Pages']==0)]['Pages']

y =  books_df[~(books_df['Pages']==0)].Price

plt.figure(figsize=(8,6))

plt.scatter(x,y,c =y,  cmap='viridis', marker='*')

plt.title("\nThe relationship between number of pages and price\n", fontdict={'fontsize':15})

plt.xlabel('The number of pages',fontdict={'fontsize':15})

plt.ylabel('Price',fontdict={'fontsize':15})

plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3)

f.set_figheight(5)

f.set_figwidth(15)

ax1.boxplot(books_df.Price)

ax1.set_xlabel("Price", fontdict={'fontsize':15})

ax2.boxplot(books_df["Publication year"])

ax2.set_xlabel("Publication year", fontdict={'fontsize':15})

ax3.boxplot(books_df.Pages)

ax3.set_xlabel("Number of pages", fontdict={'fontsize':15})

plt.show()
def correlation_heat_map(df):

    corrs = df.corr()



    # Set the default matplotlib figure size:

    fig, ax = plt.subplots(figsize=(16,12))



    # Generate a mask for the upper triangle (taken from the Seaborn example gallery):

    mask = np.zeros_like(corrs, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Plot the heatmap with Seaborn.

    # Assign the matplotlib axis the function returns. This allow us to resize the labels.

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax = sns.heatmap(corrs, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1)



    # Resize the labels.

    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14, rotation=30)

    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14, rotation=0)



    # If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.

    plt.title("\nThe correlation between  dataset features\n", fontdict = {'fontsize':20})

    plt.show()
correlation_heat_map(books_df)