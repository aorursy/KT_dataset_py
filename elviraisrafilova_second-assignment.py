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
import pandas as pd

import numpy as np

import shapefile as shp

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

data = pd.read_csv('../input/books.csv', error_bad_lines=False)
df_csv = pd.read_csv('../input/books.csv', error_bad_lines=False)
data.info()

#column name # num_pages need to be renamed. name is not appropriate

#there is no missing values in dataset
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

#it can be concluded that there is probably positive correlation between text_reviews_count and ratings_count
#first 5 rows of goodreads dataset

data.head()
#last 5 rows of goodreads dataset

data.tail()
#column names

data.columns
series = data['average_rating']        # data['Defense'] = series

print(type(series))

data_frame = data[['average_rating']]  # data[['Defense']] = data frame

print(type(data_frame))
#books with the rating higher than 4.7

x = data['average_rating']>4.7

data[x]
# number of books with average rating higher than 4.7

data[x].shape
x = data['title']

def f():

    x = data['authors']

    return x

print(x)      # x = title global scope

print(f())    # x = authors local scope
#LIST COMPREHENSİON

#high or low rating of book

threshold = sum(data.average_rating)/len(data.average_rating)

data["rating_level"] = ["high" if i > threshold else "low" for i in data.average_rating]

data.head()
#overall average rating is aquals to 3,93

overall_average_rating = sum(data.average_rating)/len(data.average_rating)

print (overall_average_rating)
data.describe()
data.boxplot(column='average_rating')
data.boxplot(column='average_rating', by ='language_code')
data_new = data.head()    

data_new
melted = pd.melt(frame=data_new,id_vars = 'title', value_vars= ['average_rating','ratings_count'])

melted
melted.pivot(index = 'title', columns = 'variable',values='value')
#first 7 rows of head and last 7 rows of tail of dataset concatinated

data1 = data.head(7)

data2= data.tail(7)

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['average_rating'].head(7)

data2= data['ratings_count'].head(7)

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
#change data types

data['rating_level'] = data['rating_level'].astype('category')

data['average_rating'] = data['average_rating'].astype('float')
data.dtypes
#check-in for missing data

data.info()

#there is no missing data in the dataset
data1 = data.loc[:,["average_rating","# num_pages","ratings_count", "text_reviews_count"]]

data1.plot()
#creating subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="ratings_count",y = "text_reviews_count")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "average_rating",bins = 50,range= (0,5),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "average_rating",bins = 50,range= (0,5),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "average_rating",bins = 50,range= (0,5),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.rating_level.value_counts()
data= data.set_index("bookID")

data.head()
data["average_rating"][4]
data.average_rating[4]
data.loc[4,["average_rating"]]
data["average_rating"] = data["average_rating"].astype('float')
data.dtypes
#why dtype shown as object?

data.loc[4,["average_rating"]]
data[["ratings_count","text_reviews_count"]]
# Difference between selecting columns: series and dataframes

print(type(data["average_rating"]))     # series

print(type(data[["average_rating"]]))   # data frames
# Slicing and indexing series

data.loc[1:20,"title":"language_code"]   
# Reverse slicing 

data.loc[20:1:-1,"title":"language_code"] 
# From something to end

data.loc[1:20,"ratings_count":] 
# Creating boolean series

boolean = data.average_rating == 5

data[boolean]
first_filter = data.average_rating == 5

second_filter = data.language_code == "eng"

data[first_filter & second_filter]
# Filtering column based others

data.title[data.average_rating<1.5]
# Plain python functions

def div(n):

    return n/2

data.average_rating.apply(div)
#rating over 100 points

data.average_rating.apply(lambda n : n*20)
# Defining column using other columns: text_reviews_count per ratings_count

data["text_reviews_per_rating"] = data.text_reviews_count / data.ratings_count

data.head()
#HIERARCHICAL INDEXING

data1 = data.set_index(["language_code","authors"]) 

data1.head(300)