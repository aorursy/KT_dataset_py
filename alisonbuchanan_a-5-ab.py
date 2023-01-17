# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization library 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#context



# I am interested in looking at airbnb price trends by date and availability. 

# Are the booked reservations on average cheaper than the available ones? 

# what do the price clusters look like over time? Do the clusters change over time?

# is there a time of year that is cheaper to book? 

# I want to understand when is the cheapest time to book
df = pd.read_csv("/kaggle/input/airbnb-nov-2019-cal/calendar.csv") #read csv file

df.head() #check the first five rows to see how data looks
df_dp = df[["date","price"]] #clean up data to isolate the two varibales we are interested in 

df_dp.head() #check the last five rows to see how the data looks
#kaggle got annoyed with me and made me include the following lines of code

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



#First,convert the columns that have numbers into numeric values so they can be easily plotted



#change the dates from string to datetime data types

x_numbers =  pd.to_datetime(df_dp['date'])



# remove the $ from the price convert price from string to float datatype 

y_numbers = df_dp["price"].replace('[\$,]', '', regex=True).astype(float)



#create a scatter plot with date and price



plt.scatter(x_numbers, y_numbers, s = 5)



#show scatter plot

plt.show()



#it is interesting to see the different pricing tiers and their consistency over time
# I was curious how a line graph would look. 

#Turns out it tells us nothing becuase there are way too many price points at each moment in time

#Pretty hilarious 



plt.plot(x_numbers, y_numbers)
#A histgram is probably more useful although I have no idea how to select the number of bins

#I am just going to play with it and increase bin size until I can see a good amount of detail 



#limit the x axis to zoom in on histogram 

plt.xlim([0, 1200])



#lable the axises for clarity

plt.ylabel('number of listings')

plt.xlabel('price ($)')



#give plot a title 

plt.title('Seattle Airbnb Prices 2019 - 2021')



#plot the histogram to see the price dsitribution accross 2000 airbnbs 

plt.hist(y_numbers, bins=3000,alpha=0.5)
#I wonder if more expensive places in Seattle are less likely to be booked? 

#first I will create a new data frame that only include booking status and price infor 



df_booked = df [["available","price"]]

df_booked
# But this df_booked data frame needs to be split

#I want to separate the free rooms (indiacted as f in the available column

#from the taken rooms (indicated as t in teh available column)

#I will store the newly split data in their own data frames to easily get each pricing averagge



#filter rows for't' which indciates the room is taken, using the boolean expression

df_t = df_booked[df_booked['available']=='t']

t_price_numbers = df_t["price"].replace('[\$,]', '', regex=True).astype(float)

df_t.tail()
# filter rows for 'f' which indciates the room is free, using the boolean expression

df_f = df_booked[df_booked['available']=='f']

f_price_numbers = df_f["price"].replace('[\$,]', '', regex=True).astype(float)

df_f.tail()
#now that the numbers have been converted to floats

#I want to find the means of taken and free airbnbs and graph them



#define names and values for bar graph

names = ['booked','available']



#lable the y axis for clarity

plt.ylabel('average price ($)')



#values are the means of both booked and availble prices

values = [t_price_numbers.mean(),f_price_numbers.mean()]



#plot the graph and show it

plt.bar(names, values)

plt.show()