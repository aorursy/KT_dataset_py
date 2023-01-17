# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2015.csv")
data.info()
data.describe()
print(data.columns)
data.head(10)
print(data.head())
#data.columns

region_list=data.Region.unique()
#print(region_list)
region_dict={}
for i in range(len(region_list)):
    region_dict[region_list[i]]=data.loc[data.Region==region_list[i]][data.columns[3]].mean()
print(region_dict)    
plot.figure(figsize=(30,10))
plot.bar(region_dict.keys(),region_dict.values())
plot.show()
    
'''
Hypothesis one: After seeing the above bar chart, we are going to assume that a country's average
happiness score is significantly affected by the region it is in. We will try to prove 
our assumption by carrying out more tests between theses two variable. We can ssume that someone
who lives in a more developed country will have a happier life. 

To furthur prove our assumption we will plot a graph of Happiness Score vs GDP in the next cell
'''    


#Scatter graph for comparing the GDP per capita and the Happiness Score
plot.scatter(data[data.columns[5]],data[data.columns[3]])
plot.xlable="GDP per Capita"
plot.ylable="Happiness Score"
print(np.corrcoef(data[data.columns[5]],data[data.columns[3]]) [1,0])

plot.show()
'''
As we can see, there is mediocre positive correlation between the GDP per capita and
the Happiness of a country, with a perason's corrolation coefficent of 0.780965526866
To furthur prove our Hypotheisis one:
1.we will next plot a bar chart of the regions 
and the average GDP of the region.
2. Create a scatter plot of mean happiness score and mean GDP per capita by region
'''
'''
Section for plotting the bar chart between average GDP per capita by region and region
'''