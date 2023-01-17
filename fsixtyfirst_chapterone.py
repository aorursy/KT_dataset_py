import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns  





megdata =pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

megdata.head(7)
megdata.info()
megdata.corr()

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(megdata.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.show()
megdata.budget.plot(kind = 'line', color = 'b',label = 'Budget',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')

megdata.revenue.plot(color = 'r',label = 'Revenue',linewidth=1, alpha = 0.7,grid = True,linestyle = '-.')

plt.legend(loc='upper left')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot') 

plt.show()
megdata.plot(kind='scatter', x='budget', y='revenue',alpha = 0.5,color = 'red')

plt.xlabel('budget')              

plt.ylabel('revenue')

plt.title('Budget Revenue Scatter Plot')           

plt.show()
megdata.budget.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
megdata.popularity.plot(kind = 'hist',bins = 50)

plt.clf()

averagerevenue = sum(megdata.revenue) / len(megdata.revenue)

print(int(averagerevenue))

megdata['money'] = ["gayet iyi" if i > averagerevenue else "kotü" for i in megdata.revenue ]

print(megdata.money)







# 1 -I calculated the average speed here 



# 2 -I wrote 'fine' to those who more than 'averagerevenue' and wrote 'bad' to those who less than 'averagerevenue'