# Import Necessary Libraries and environment for Analysis and visualization



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

# Seaborn Library

import seaborn as sns



# See the data sets and their path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the data set file into a  pandas data frame

df = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv",encoding="ISO-8859-1")
df.head()
df.info()
print("Number of columns/Features :", df.shape[1])

print("Number of rows/observations :", df.shape[0])
# Any blanks and NAs in the data ?

check = df.isnull().values.any()

if check == True:

    print(" There are Null values  in the data frame")

else:

    print(" The data seems to have no null values ")

df['state'].value_counts()
# Remove the rows that are having number as 0.0 

df = df[df['number'] != 0]

df.head()
sns.set(rc={'figure.figsize':(25,10)})

set1 = df[['state', 'number']]

set1 = set1.groupby('state').sum()

#set1.reset_index(inplace = True)

set1 = set1.sort_values('number', ascending=False)

set1.reset_index(inplace = True)

ax = sns.barplot(x="state", y="number", data=set1)

for index, row in set1.iterrows():

    ax.text(x = row.name, y = row.number, s = str(round(row.number)),color = 'black',ha = 'center')

ax.set_title("State Wise Forest Fires")    
sns.set(rc={'figure.figsize':(25,10)})

set2 = df[['year', 'number']]

set2 = set2.groupby('year').sum()

set2.reset_index(inplace = True)

ax = sns.barplot(x="year", y="number", data=set2)

for index, row in set2.iterrows():

    ax.text(x = row.name, y = row.number, s = str(round(row.number)),color = 'black',ha = 'center')

ax.set_title("Year Wise Forest Fires") 
set2 = df[['month','number']]

set2 = set2.groupby('month').mean()

set2.reset_index(inplace = True)

ax = sns.barplot(x="month", y="number", data=set2)

for index, row in set2.iterrows():

    ax.text(x = row.name, y = row.number, s = str(round(row.number)),color = 'black',ha = 'center')

ax.set_title("All years, Monthly average Forest Fires")    

plt.show()
from pylab import rcParams

import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 30, 10



yearly = df[['year', 'number']]

yearly = yearly.groupby('year').sum()



yearly.reset_index(inplace = True)

# Plotting using the matplot lib

#make x , y parameters

x = yearly['year'] ; y = yearly['number']

plt.plot(x,y)

plt.title("Yearly Wild Fires ", fontsize=40)

plt.xlabel("Years", fontsize = 30)

plt.ylabel('Number', fontsize = 30)

plt.gca().set_xticks(yearly["year"].unique())

plt.show()

# pick each year Max and min 

MAX_MONTH = []; MAX_CNT = []; MIN_MONTH = []; MIN_CNT = []

years = list(df['year'].unique())

years.sort()

for x in years:

    data_set = df[df['year']== x]

    data_set = data_set[['month', 'number']]

    data_set = data_set.groupby('month').mean()

    data_set.reset_index(inplace = True)

    data_set_max = data_set[data_set['number'] == max(data_set['number'])]

    MAX_MONTH.append(list(data_set_max['month'])[0])

    MAX_CNT.append(list(data_set_max['number'])[0])

    data_set_min = data_set[data_set['number'] == min(data_set['number'])]

    MIN_MONTH.append(list(data_set_min['month'])[0])

    MIN_CNT.append(list(data_set_min['number'])[0])
# Prepare the data set with Max , Min  Numbers and occuring Months

max_min_set = pd.DataFrame(years)

max_min_set['MAX_MONTH'] =list(MAX_MONTH)

max_min_set['MAX_CNT'] =list(MAX_CNT)

max_min_set['MIN_MONTH'] =list(MIN_MONTH)

max_min_set['MIN_CNT'] =list(MIN_CNT)
max_min_set
max_min_set[['MAX_MONTH', 'MAX_CNT']].groupby('MAX_MONTH').mean().plot(kind = 'bar')

plt.title("Maximum Fires occuring Months", fontsize = 30)

plt.xlabel("Maximum Fire Months",fontsize = 30)

plt.ylabel("Fire Counts",fontsize = 30)

plt.show()
max_min_set[['MIN_MONTH', 'MIN_CNT']].groupby('MIN_MONTH').mean().plot(kind = 'bar')

plt.title("Minimum Fires occuring Months", fontsize = 30)

plt.xlabel("Minimum Fire Months",fontsize = 30)

plt.ylabel("Fire Counts",fontsize = 30)

plt.show()