

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import datasets

from sklearn.cluster import KMeans

import sklearn.metrics as sm



from pandas import DataFrame



# Set some pandas options

pd.set_option('display.notebook_repr_html', False)

pd.set_option('display.max_rows', 60)

pd.set_option('display.max_columns', 60)

pd.set_option('display.width', 1000)

 

%matplotlib inline



# read csv file 

df = pd.read_csv("../input/books.csv", error_bad_lines=False)



df.head()
print("Row: ",df.shape[0])

print("Column: ",df.shape[1])
df.hist()

plt.show()
correlations = df.corr()

names = ['#num_pages','average_rating','#ratings_count','#text_reviews_count']

# plot correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,5,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
df1=df.drop(columns=['bookID','# num_pages', 'isbn','isbn13','ratings_count','text_reviews_count'])

#df_bookrating=df['average_rating']

df_worst_bookrating=df1.sort_values("average_rating", ascending=True)

#print("Row: ",df1.shape[0])

#print("Column: ",df1.shape[1])

#df2=df['authors']

df_worst_bookrating.head(10)

# create new data frame based on my selection 

df_book=df[['language_code','authors','average_rating']]



# high rating books 

df_hrating=df_book[df_book.average_rating> 4.8]



# only english books

df_bookselection=df_book[df_book["language_code"]=="eng"]

# print top 10 values 

df_bookselection.head(10)


