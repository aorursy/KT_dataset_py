# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#reading the csv using pandas
#in R 'diamonds.csv' is part of the ggplot2 library you would have to import the library and then access the 
#datasets that comes with the library using data()
df = pd.read_csv('../input/diamonds.csv')
#initial look at the dataset
#in R you would need to insert the name of the dataset as the first argument of the head() function
df.head()
#shape of our dataset
# in R you would insert the name of the dataset as the first argument of the dim() function 
df.shape
#removing the first column of the dataset using iloc
#you do not need to do this for the diamonds dataset in R. An easy way to do this in R is to set the column 
#to null
#diamomnds[1] <- NULL
df = df.iloc[:,1:]
#The first thing I would like to do is check if there are any 'Null' values in the dataset 
#in R a way to check if there are na values in the columns of our dataset is using this function 
#apply(diamonds, 2, function(x) any(is.na(x)))
df.isnull().any()
#explore the dataset by seeing how cut and color of a diamond affect the price
# histogram of price counts faceted by color and the cut of the diamond is used to color the histogram bars.
p1 = sns.FacetGrid(data = df, col ='color', col_wrap = 4, hue = 'cut')
(p1.map(plt.hist, 'price')).add_legend();

#to do this in R
#ggplot(aes(x = price), data = diamonds)+
#  geom_histogram(aes(fill = cut))+
#  facet_wrap(diamonds$color)
#I think it is more visually appealing and easier to read the output in R
# A scatterplot of table vs. price
sns.lmplot(x='table', y='price', data=df, palette = 'Set1',
           fit_reg=False, # No regression line
           hue='cut', #color of the markers by cut
           scatter_kws={"s": 5, 'alpha':1}) #size of the markers and the alpha

plt.xlim(50,80); #setting the min and max limit for the x axis

#ggplot(aes(x=table, y= price), data = diamonds)+
#  geom_point(aes(color = cut))+
#  scale_x_continuous(breaks = seq(50,80,2), limit = c(50,80))
#let us now use the x,y and z to calculate the volume of the diamonds
# in R to create the new volume column you could
#diamonds$volume <- diamonds$x * diamonds$y * diamonds$z
df['volume'] = df.x * df.y *df.z

#create a scatterplot of the price vs volume to see how the volume affects the price for diamonds of different
#clarity
p3 = sns.lmplot(data = df, x = 'volume', y = 'price', fit_reg = False, hue = 'clarity', scatter_kws = {'s':5})

plt.xlim(0,df.volume.quantile(0.99));

#to do this in R
#ggplot(aes(x = volume, y = price), data = diamonds)+
#  geom_point(aes(color = cut))+
#  scale_y_log10()+
#  xlim(0,quantile(diamonds$volume,0.99))



