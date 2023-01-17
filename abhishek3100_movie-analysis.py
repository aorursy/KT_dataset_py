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
df = pd.read_csv("../input/tmdb_movies_data.csv")

df.info()
df.head()
df.tail(2)
df.describe()
#this dataset contains null value also 

#lets find out the null value

df.isna().sum()
#fill the null values with 0

df.fillna(0)

#finding duplicate rows

sum(df.duplicated())
#Dropping duplicate row

df.drop_duplicates(inplace = True)

#After Dropping

print("After Dropping Table(Rows, Col) :", df.shape)
#Changing Format Of Release Date Into Datetime Format

df['release_date'] = pd.to_datetime(df['release_date'])
df['release_date'].head()
#lets remove these coloumn

df.drop(['budget_adj','revenue_adj','overview','imdb_id','homepage','tagline'],axis =1,inplace = True)
print("(rows,cols): ",df.shape)
#checking with 0 values in budget and revenew columns

print("Budget col having 0 value :",df[(df['budget']==0)].shape[0])

print("Revenue col having 0 value :", df[(df['revenue']==0)].shape[0])
import seaborn as sns

import matplotlib.pyplot as plt
# Counting the number of movies in each year 

data = df.groupby('release_year').count()['id']

print(data.tail())
#grouping the data a/q to release year and counting movies



sns.set(rc={'figure.figsize':(10,5)})

sns.set_style("darkgrid")

df.groupby('release_year').count()['id'].plot(xticks = np.arange(1960,2016,5))

plt.title("Year Vs Number Of Movies",fontsize = 14)

plt.xlabel('Release year',fontsize = 13)

plt.ylabel('Number Of Movies',fontsize = 13)
df['Profit'] = df['revenue'] - df['budget']

def find_minmax(x):

    #use the function 'idmin' to find the index of lowest profit movie.

    min_index = df[x].idxmin()

    #use the function 'idmax' to find the index of Highest profit movie.

    high_index = df[x].idxmax()

    high = pd.DataFrame(df.loc[high_index,:])

    low = pd.DataFrame(df.loc[min_index,:])

    

    #print the movie with high and low profit

    print("Movie Which Has Highest "+ x + " : ",df['original_title'][high_index])

    print("Movie Which Has Lowest "+ x + "  : ",df['original_title'][min_index])

    return pd.concat([high,low],axis = 1)



#call the find_minmax function.

find_minmax('Profit')
#make a plot which contain top 10 movies which earn highest profit.

#sort the 'Profit' column in decending order and store it in the new dataframe,

info = pd.DataFrame(df['Profit'].sort_values(ascending = False))

info['original_title'] = df['original_title']

data = list(map(str,(info['original_title'])))

x = list(data[:10])

y = list(info['Profit'][:10])



#make a plot usinf pointplot for top 10 profitable movies.

ax = sns.pointplot(x=y,y=x)



#setup the figure size

sns.set(rc={'figure.figsize':(10,5)})

#setup the title and labels of the plot.

ax.set_title("Top 10 Profitable Movies",fontsize = 15)

ax.set_xlabel("Profit",fontsize = 13)

sns.set_style("darkgrid")
df['budget'] = df['budget'].replace(0,np.NAN)

find_minmax('budget')