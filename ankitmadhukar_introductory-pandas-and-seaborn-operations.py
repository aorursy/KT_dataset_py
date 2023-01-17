# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df =  pd.read_csv('../input/Pokemon.csv')  #read the csv file and save it into a dataframe
df.head(n=10) 
## We are going to work with the legendary pokemons . So we have to filter them out 
df = df[df['Legendary']==True]
## We use the head function to visualize the new dataframe
df.head(10)
df.describe()
## We drop the # column 
df=df.drop(['#'],axis=1)

#set the name as the index
df = df.set_index('Name')
df.head(5)
## We now remove the extra text with the Mega Pokemons. Code from learn-pandas-with-python 
df.index = df.index.str.replace(".*(?=Mega)", "")
df.head(10)

## Lets find the pokemons which are the strongest in our dataset . We can do so by using the Total column to sort the df in descending 
## order and print the head to find the pokemons. For eg we want to find the 5 strongest pokemons.

df.sort_values('Total',ascending=False).head(5)
## WE see that they are almost equally powerful . What about the weakest of the lot.
df.sort_values('Total',ascending=True).head(3)
##  ascending parameter is used to sort the list in ascending or descending order
## Want to visualize some data ? seaborn is a great library to do so
import seaborn as sns
x = df.loc[:,"Attack"]
y = df.loc[:,"Defense"]


sns.jointplot(x=x, y=y, data=df, kind="kde");
sns.pairplot(df) ## This creates a matrix of axes and shows the relationship for each pair of columns in a DataFrame.
##by default, it also draws the univariate distribution of each variable on the diagonal Axes
## Which Pokemon has the highest HP 
x= df.loc[df["HP"].idxmax()]
x

## Whats the realation between attacking power and the pokemon type ? Lets check
## We use the second type as hue to make the data more informative
import matplotlib.pyplot as plt
type1 = df.loc[:,'Type 1']
Attack = df.loc[:,'Attack']
plt.subplots(figsize=(10,8))
sns.swarmplot(x=type1, y=Attack, hue="Type 2", data=df);
##Lets plot the distribution of all the numerical values . In doing so we drop The following columns from the dataframe
df1 = df.drop(columns=['Legendary','Generation','Total'])
plt.subplots(figsize=(10,8))
order = df1.std().sort_values().index
sns.lvplot(data=df1, order=order, scale="linear", palette="mako")
## Hope you liked my work Thank You - Ankit Madhukar