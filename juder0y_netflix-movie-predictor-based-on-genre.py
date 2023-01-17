# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nf = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

nf.head()
movie = nf[nf.type == "Movie"][['show_id','title','listed_in']]

movie.info()

# movie['listed_in'].count('Comedies')

# nf["listed_in"].nunique()
movie['listed_in'] = movie['listed_in'].str.replace(" ","")

genrelist = movie['listed_in'].tolist() #create a list of all entries in listed_in

len(genrelist) #length of list
i=0

temp = []

while i < 4265: # iterating through each object to fetch indedependent genres 

    temp = temp + genrelist[i].split(',')

    i = i+1   

genreset = set(temp) # creating a set get unique genres

uniquegenre = list(genreset) # created a list of unique genres

for j in uniquegenre: #making columns for each genre and assigning value 0

    movie[j] = 0

movie.info()    

for j in uniquegenre: #iterating through unique genres

    movie[j] = movie.apply(lambda x: int(j in x.listed_in),axis = 1) #lambda function checks if current genre is in listed_in entry

movie.head(20)    
import scipy.cluster.hierarchy as shc

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(movie.iloc[:,3:23], method='ward'))
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')  

movie['cluster'] = cluster.fit_predict(movie.iloc[:,3:23])
movie[movie.cluster == 8][['title','listed_in','cluster']].head(20)    
movie.listed_in.tolist()

movieplt = nf[nf.type == 'Movie'][['release_year','rating']]

movieplt = movieplt.fillna("not known")

plt.figure(figsize=(10, 7))  



plt.scatter(movieplt.release_year, movieplt.rating, c=cluster.labels_,cmap='rainbow')
movieplt = nf[nf.type == 'Movie'][['release_year','rating']]

movieplt