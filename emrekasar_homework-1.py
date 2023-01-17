import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df.head(15) #To show the first 5 elements and to see if I can demonstrate my csv
df.info()
df.corr()
f,ax = plt.subplots(figsize=(8, 8)) 

sns.heatmap(df.corr(),cmap = "coolwarm", annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
df.sort_values(by ="id").head(10) #I was just being curious to sort my dataframe.
profile = ProfileReport(df, title = "Profiling Report", html = {'style':{'full_width':True}}) #It took some time to load
profile.to_widgets() #Also, it took some time to load
df.popularity.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')

df.runtime.plot(kind = 'line', color = 'b',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Films')              # label = name of label

plt.ylabel('Popularity,Runtime')

plt.title('Line Plot')            # title = title of plot

plt.show()
df.plot(kind = "scatter", x = "popularity" , y = "runtime" , alpha = 0.5, color = "purple", grid = True)

plt.title("Popularity - Runtime")

plt.show()
counter = 0

for i in df.revenue:

    if(i%2 == 0):

        counter +=1

print(counter)

#Prints the number of revenues which are even.