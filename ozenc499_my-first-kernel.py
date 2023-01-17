import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.info()
data.corr(method ='kendall') #lets try kendall
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(method ='kendall'), annot=True, linewidths=.5, fmt= '.1f',cmap="YlGnBu", ax=ax, cbar_kws={"orientation": "horizontal"})

plt.show()
data.head()
data.tail()
data.head().append(data.tail())     #I think Elisabeth and John paied a lot just for a room  :) :)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.price.plot(kind = 'line', color = 'b',label = 'price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.calculated_host_listings_count.plot(color = 'r',label = 'listings',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = number_of_reviews, y = reviews_per_month

data.plot(kind='scatter', x='reviews_per_month', y='number_of_reviews',alpha = 0.5,color = 'red')

plt.xlabel('reviews_per_month')              # label = name of label

plt.ylabel('number_of_reviews')

plt.title('Reviews Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.availability_365.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.xlabel('availability_365')  

plt.show()
dictionary = {'Netherlands' : 'Feyenoord'}

print(dictionary.keys())

print(dictionary.values())
# Adding a new entry to dictionary

dictionary['France'] = "Nantes" 

print(dictionary)

print(dictionary.keys())

print(dictionary.values())
# Update an entry in the dictionary

dictionary['Netherlands'] = "Ajax" 

print(dictionary)
# Delete an entry 

del dictionary['France']  

print(dictionary)
# How to check an entry in a dictionary

print('France' in dictionary) 
# remove all entries in a dictionay

print(dictionary)

dictionary.clear()

print(dictionary)

series = data['price']

print(series)

print(type(series))

data_frame = data[['price']]

print(data_frame)

print(type(data_frame))
print(series[0])

print(series[1])
print(series[0] > series[1])

print(series[0] != series[1])
#lets do some filtering exercises



expensive = data['price'] > 9999   # This is a series

print('\n')

print(type(expensive))

print('\n')

print('Below is a series\n')

print(expensive)

print('\n')

print('You can filter any data by using a series in the dataframe;\n' )

new_data_frame= data[expensive]

print('Please see the type first; ', type(new_data_frame))

print('\n')

print('And here is the filtered dataframe;\n')

print(new_data_frame)

print('\n')

print('These are the hosts who has very expensive place for rent\n')

print(new_data_frame['host_name'])

# Filtered dataframe output is shown as in the below;

data[expensive]
# usage of logical_and ( a numpy function)



data[np.logical_and(data['price']>5000, data['room_type']== "Private room" )]
# Stay in loop if condition

i = 0

while i < 5 :

    i = i + 1

    print('{}. while loop'.format(i))

print('\n')

 

    

list = [1,2,3,4,5]

for i in list:

    print('i is: ',i)

print('\n')