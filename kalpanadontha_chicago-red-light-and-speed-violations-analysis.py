import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

#There are 3 csv files in the current version of the dataset:

redlightcam = pd.read_csv('../input/red-light-camera-locations.csv')

redlightvoilation = pd.read_csv('../input/red-light-camera-violations.csv')

#speedvoilations = pd.read_csv('../input/speed-camera-violations.csv')
print(os.listdir('../input'))
#nRowsRead = 100 # specify 'None' if want to read whole file

# red-light-camera-locations.csv has 149 rows in reality, but we are only loading/previewing the first 100 rows

#redlightcam = pd.read_csv('../input/red-light-camera-locations.csv')

#redlightcam = pd.read_csv('../input/red-light-camera-locations.csv', delimiter=',', nrows = nRowsRead)

# redlightcam.dataframeName = 'red-light-camera-locations.csv'

nRow, nCol = redlightcam.shape

print(f'There are {nRow} rows and {nCol} columns')
redlightcam.head(15)
#nRowsRead = 100 # specify 'None' if want to read whole file

# red-light-camera-violations.csv has 395227 rows in reality, but we are only loading/previewing the first 100 rows

# redlightvoilation = pd.read_csv('../input/red-light-camera-violations.csv', delimiter=',', nrows = nRowsRead)

# redlightvoilation.dataframeName = 'red-light-camera-violations.csv'

nRow, nCol = redlightvoilation.shape

print(f'There are {nRow} rows and {nCol} columns')
redlightvoilation.head(5)
#nRowsRead = 100 # specify 'None' if want to read whole file

# speed-camera-locations.csv has 155 rows in reality, but we are only loading/previewing the first 100 rows

#speedvoilations = pd.read_csv('../input/speed-camera-locations.csv', delimiter=',', nrows = nRowsRead)

# speedvoilations = pd.read_csv('../input/speed-camera-locations.csv', delimiter=',', nrows = nRowsRead)

# nRow, nCol = speedvoilations.shape

# print(f'There are {nRow} rows and {nCol} columns')
#Seaborn

#Barplot-A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent.

#PairPlot- shows all numerical variables paired with all the other numerical variables in a Dataframe(relationship between two variables )

#Catplot - It shows/reporesents the catorization of the the 2 variables(relationship between a numerical and one or more categorical variables)
#redlightcam['GO LIVE DATE'][0]

sns.pairplot(redlightcam)
#Checking if any null values in Intersection column

redlightvoilation['INTERSECTION'].isnull().sum()
#Showing the Intection count( No NAN values)

redlightvoilation['INTERSECTION'].value_counts()
#PairPlot- shows all numerical variables paired with all the other numerical variables in a Dataframe(relationship between two variables )

sns.pairplot(redlightvoilation,x_vars = 'VIOLATIONS', y_vars='CAMERA ID')
#Catplot - It shows/reporesents the catorization of the the 2 variables(relationship between a numerical and one or more categorical variables)

sns.catplot(x="VIOLATIONS", y="CAMERA ID",hue='VIOLATIONS',jitter=False, data=redlightvoilation)
#Plotting a bar graph - No of Voilations / intersection



intersection = redlightvoilation['INTERSECTION'].value_counts()

intersection = intersection[:10]

# print(intersection)

# print(intersection.values)

# print(intersection.index)

plt.figure(figsize=(10,5))

sns.barplot(x=intersection.index.str.replace(' ', '\n'),y=intersection.values,alpha=0.8)

plt.title(' Top 10  Violations in Chicago at Different Intersections')

plt.ylabel('Number of Occurrences', fontsize=10)

plt.xlabel('Intersections', fontsize=10)

plt.xticks(rotation=308)

plt.show()
# Plottng a bar graph showing the no of voilation / year.

# From the analysis we can conclude te 2016 has seen more violations in the last 4 years, but in general,the viloations have increased over a period of time.



redlightvoilation['YEAR'] = redlightvoilation['VIOLATION DATE'].apply(lambda date:pd.Period(date, freq='Q').year)

redlightvoilation.groupby('YEAR')['VIOLATIONS'].sum()

x_axis = (redlightvoilation.groupby('YEAR')['VIOLATIONS'].sum())

y_axis = (redlightvoilation.groupby('YEAR')['VIOLATIONS'].sum())

plt.figure(figsize=(10,5))

sns.barplot(x=x_axis,y=y_axis,alpha=0.8)

plt.title(' Number of Violations / Year')

plt.ylabel('Violatons', fontsize=10)

plt.xlabel('Year', fontsize=10)

plt.show()