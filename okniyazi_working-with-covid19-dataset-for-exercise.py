# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#We are using the pandas library to read data that are in csv format.

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
# head() function shows us the first 5 data rows that are in our dataset.

data.head(10) # it shows 10 rows 
# We can check column names with 'columns' function

data.columns = ['SNo','ObservationDate', 'State', 'Country', 'Last_Update', 'Confirmed', 'Deaths','Recovered'] # We changed column names for easy use.

data.columns
#we are using this type of plots for correlations.

f,ax = plt.subplots(figsize  = (12,12))

sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.1f', ax = ax) #annot = writes the correlation stat inside a cell.

#ax = matplotlib Axes

# fmt = number of digits after '.'

#for more info visit --> https://seaborn.pydata.org/generated/seaborn.heatmap.html

plt.show()
#Confirmed cases and deaths.

data.Confirmed.plot(kind = 'line', color = 'cyan', label = 'Confirmed', linewidth = 1, alpha = .5, grid = True, linestyle = '--')

data.Deaths.plot(color = 'red', label = 'Deaths', linewidth = 1, alpha = .5, grid = True, linestyle = ':')

plt.legend(loc = 'upper left')

plt.xlabel('X') 

plt.ylabel('Y')

plt.title('Covid19 Dataset Line Plot')

plt.show()
#Countries that have more than 10.000 deaths because of Covid19.

dangerous_countries = data['Deaths'] > 10000 # filtering our data.

data[dangerous_countries].plot(kind = 'scatter', x = 'Country', y = 'Deaths', alpha = .5, color = 'red', figsize = (15,10))

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('Covid19 Dataset Scatter Plot')
#

united_kingdom_data = data['Country'] == 'UK' #filtering our data.

data[united_kingdom_data].Recovered.plot(kind = 'hist', bins=100, figsize = (20,15))

plt.xlabel('Recovery')

plt.ylabel('Frequency')

plt.title('Covid19 Dataset Histogram Plot about UK')

plt.show()
#We can read data that are in csv format as we did before.

###data = pd.read_csv('...../datapath')
series = data['Deaths']        # data['Deaths'] = series

print(type(series))

data_frame = data[['Deaths']]  # data[['Deaths']] = data frame

print(type(data_frame))
#Let's see Turkey's data about Covid19 with filtering.

turkey_data = data['Country'] == 'Turkey'

data[turkey_data].head(20)
#We can use logical_and // logigal_or functions which comes from numpy library.

#Let's see which countries have more than 100.000 confirmed case and over 30.000 dead.

dangerous = np.logical_and(data['Confirmed'] > 100000, data['Deaths'] > 30000)

data[dangerous]

#We can do this operation with bitwise operators like --> data[(data['Confirmed']>100000) & (data['Deaths']>30000)] 
#For Loop

#Let's say we have list which includes random numbers and we are trying to find them.

list = [13,85,97]



for i in list :

    print('I find number in the list: ',i) 

#While Loop

#Let's say we are trying to make countdown.

i = 10



print('The Big Countdown')



while i != 0 :

    print('Last', i ,"Second!")

    i -= 1

print('Happy New Year!!!')    

#Let's make one more exercise about loops with using dictionary.



dictionary = {'apple':'red','banana':'yellow', 'cucumber':'green'}

print('Fruit  - Color')

for key,value in dictionary.items():

    print(key," : ",value)

print('')
