#references 

#  https://www.kaggle.com/yashvi/window-store-analysis
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import tensorflow as tf 

import io

from matplotlib import style 

style.use('ggplot')
filename  = '../input/windows-store/msft.csv'

df = pd.read_csv(filename)
df.head()
df.index = df.Date

df.head()
# As most of the values are Free in the price column we need to set it to zero 

for i in range(len(df)):

    if df.Price[i]=='Free':

        df.Price[i]=0.0

df.head()
#Plotting the mean rating of the categories 

category_array = df.Category.unique()

mean_rating_list =[]

for i in range(len(category_array)):

    df1 =  df[df['Category']==category_array[i]]

    mean_rating = df1['Rating'].mean()

    mean_rating_list.append(mean_rating)

# mean_rating_list



index =[ x for x in range(len(category_array))]



plt.bar(index,mean_rating_list)

plt.xticks(index,category_array,rotation=90)

plt.xlabel("Category_of_Books")

plt.ylabel("Average_Rating")

plt.plot()

#calulating which category of books whom were rated by most no of people 

category_array = df.Category.unique()

no_of_rated =[]

for i in range(len(category_array)):

    df1 =  df[df['Category']==category_array[i]]

    rate_count= df1['No of people Rated'].sum()

    no_of_rated.append(rate_count)



index =[ x for x in range(len(category_array))]



plt.bar(index,no_of_rated)

plt.xticks(index,category_array,rotation=90)

plt.xlabel("Category_of_Books")

plt.ylabel("No_of_rating")

plt.plot()

# getting the rating over the years 

import datetime as dt 



df['Date'] = pd.to_datetime(df['Date'])

df.Date.dtype

#Plot to visualize how ratings fluctutated over the years.



def plot_rating_over_years():

    df['Date'].sort_values()

    df1 = df.groupby(df['Date'].dt.year)['Rating'].mean()

    df1 = pd.DataFrame(df1)

    y = df1.Rating.tolist() 

    x = df1.index 



    #plotting the average rating over the years 

    plt.plot(x,y)

    plt.xlabel('Years')

    plt.ylabel('Rating')

    plt.show()



plot_rating_over_years()
#Plotting the number of ratings per year.

def plot_numpeople_rating():

    df1= df.groupby(df['Date'].dt.year)['No of people Rated'].count()

    # df1

    df1 = pd.DataFrame(df1)

    y = df1['No of people Rated'].tolist() 

    x = df1.index 



    #plotting the average rating over the years 

    plt.plot(x,y)

    plt.xlabel('Years')

    plt.ylabel('No of people Rated')

    plt.show()





plot_numpeople_rating()
from wordcloud import WordCloud
Category=df['Category'][~pd.isnull(df['Category'])]

wordCloud = WordCloud(width=350,height= 250).generate(' '.join(Category))

plt.figure(figsize=(19,9))

plt.axis('off')

plt.title(df['Category'].name,fontsize=20)

plt.imshow(wordCloud)

plt.show()

# This is my first notebook, 

# any suggestions would be appreciated.