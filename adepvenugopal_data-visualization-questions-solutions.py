#Import pandas package as pd

import pandas as pd
#import matplotlib.pyplot package as plt 

import matplotlib.pyplot as plt 
#Import the CSV file winemag-data_first150k.csv into wine_data (do not include index column)

#wine_data = pd.read_csv("<Enter the full path here>", index_col=0)

wine_data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
#Display first 5 lines of wine_data using head() function

wine_data.head()
#Find value_counts of province column

#__._____.value_counts()

wine_data.province.value_counts()
#Find value_counts() of province column and display first 10 rows using head() function

#__._____.value_counts().head(__)

wine_data.province.value_counts().head(10)
#Find value_counts() of province column and display first 10 rows using head() function and plot the BAR graph

#__._____.value_counts().head(__).plot.bar()

wine_data.province.value_counts().head(10).plot.bar()
#Plot the figure in the 18 x 6 inches grid

#plt.figure(figsize=(___,___))

plt.figure(figsize=(18,6))

wine_data.province.value_counts().head(10).plot.bar()
#Find value_counts() of province column and display first 10 rows, normalize the data and plot the BAR graph

#(__._____.value_counts().head(__)/len(wine_data).plot.bar()

plt.figure(figsize=(18,6))

(wine_data.province.value_counts().head(10)/len(wine_data)).plot.bar()
#Find value_counts of points column

#__._____.value_counts()

wine_data.points.value_counts()
#Find value_counts of points column and sort by index

#__._____.value_counts().sort_index()

wine_data.points.value_counts().sort_index()
#Find value_counts of points column and sort by index and plot the BAR graph

#__._____.value_counts().sort_index().plot.bar()

plt.figure(figsize=(18,6))

wine_data.points.value_counts().sort_index().plot.bar()
#Find value_counts of points column and sort by index and plot the LINE graph

#__._____.value_counts().sort_index().plot.line()

plt.figure(figsize=(18,6))

wine_data.points.value_counts().sort_index().plot.line()
#Find value_counts of points column and sort by index and plot the AREA graph

#__._____.value_counts().sort_index().plot.area()

plt.figure(figsize=(18,6))

wine_data.points.value_counts().sort_index().plot.area()
#Display the rows of wine_data where the price > 200

#__[________['________']>200]

wine_data[wine_data['price']>200]
#Display the rows of wine_data where the price > 200.  Display only price column

#__[________['________']>200]['------']

wine_data[wine_data['price']>200]['price']
#Display the rows of wine_data where the price > 200.  Display only price column and plot the histogram.

#__[________['________']>200]['------'].plot.hist()

plt.figure(figsize=(18,6))

wine_data[wine_data['price']>200]['price'].plot.hist()
#Display the rows of wine_data where the price > 200.  Plot the histogram of price with 100 bins

#__[________['________']>200]['------'].plot.hist(bins=100)

plt.figure(figsize=(18,6))

wine_data[wine_data['price']>200]['price'].plot.hist(bins=100)
#Display the rows of wine_data where the price > 1500.  Plot the histogram of price with 100 bins

#__[________['________']>1500]['------'].plot.hist(bins=100)

plt.figure(figsize=(18,6))

wine_data[wine_data['price']>1500]['price'].plot.hist(bins=100)
#Display value_counts() of country in wine_data. Display first 5 rows only.

#__._____.value_counts().head()

wine_data.country.value_counts().head()
#Display value_counts() of country in wine_data. Display first 5 rows only. Plot the pie chart

#__._____.value_counts().head().plot.pie()

plt.figure(figsize=(8,8))

wine_data.country.value_counts().head().plot.pie()
#Display the rows of wine_data where the price < 100.  

#_______[________['________']<100]

wine_data[wine_data['price'] < 100]
#Display the rows of wine_data where the price < 100.  Display 5 random samples

#_______[________['________']<100].sample(__)

wine_data[wine_data['price'] < 100].sample(50)
#Display the rows of wine_data where the price < 100.  Plot the scatter plot with 'price' on x axis, 'points' on y axis for 50 random samples

#_______[________['________']<100].sample(__).plot.scatter(x='------',y='------')

wine_data[wine_data['price'] < 100].sample(50).plot.scatter(x='price',y='points')
#Display the rows of wine_data where the price < 100.  Plot the scatter plot with 'price' on x axis, 'points' on y axis for all samples

#_______[________['________']<100].plot.scatter(x='------',y='------')

wine_data[wine_data['price'] < 100].plot.scatter(x='price',y='points')
#A hex plot aggregates points in space into hexagons, and then colors those hexagons based on the values within them

#Display the rows of wine_data where the price < 100.  Plot the hexplot with 'price' on x axis, 'points' on y axis and gridsize 15.

#wine_data[wine_data['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)

wine_data[wine_data['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
#Display the rows of wine_data where the price > 100.  Plot the histogram of price with 100 bins

#__[________['________']>1500]['------'].plot.hist(bins=100)

wine_data[wine_data['price'] < 100].sample(100).plot.scatter(x='price', y='points')
#Import the CSV file winemag-data_first150k.csv into wine_data (do not include index column)

#wine_data = pd.read_csv("<Enter the full path here>", index_col=0)

wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv",index_col=0)
#Display first 5 rows of wine_counts

wine_counts.head()
#wine_counts counts the number of times each of the possible review scores was received by the five most commonly reviewed types of wines:

#Plot the stacked BAR chart of wine_counts

#_______.plot.bar(stacked=True)

wine_counts.plot.bar(stacked=True)
#Plot the stacked AREA chart of wine_counts

#_______.plot.area(stacked=True)

wine_counts.plot.area(stacked=True)
#Plot the stacked LINE chart of wine_counts

#_______.plot.line(stacked=True)

wine_counts.plot.line(stacked=True)
#Import the Pokemon.csv" dataset

#pokemon = pd.read_csv("_______________________", index_col=0)

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
#Display first 5 rows of pokemon dataset

#________.head()

pokemon.head()
#Plot scatter plot with 'Attack' on x-axis and 'Defence' on y-axis

#pokemon.plot.scatter(x = '_________', y = '_________')
#A hex plot aggregates points in space into hexagons, and then colors those hexagons based on the values within them

#Plot scatter plot with 'Attack' on x-axis and 'Defence' on y-axis with grid size of 15

#pokemon.plot.hexbin(x='__________', y='___________', gridsize=___)

pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=15)
#Plot the bar chart of points column of wine_data

wine_data['points'].value_counts().sort_index().plot.bar(figsize=(12, 6))