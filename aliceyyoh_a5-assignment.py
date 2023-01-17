# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the file that I uploaded in Kaggle Data
dataset = pd.read_csv("../input/a5data/NCHS_-_Leading_Causes_of_Death__United_States.csv")
df = pd.DataFrame(dataset)

# read the table
df
# the column '113 Cause Name' and 'Cause Name' are almost indifferent, so I'd like to delete the column '113 Cause Name'

# use drop method
df = df.drop(columns=['113 Cause Name'])
# change the name of index "Age-adjusted Death Rate" to make it more concise and understandable

df = df.rename(columns={'Age-adjusted Death Rate': 'Death Rate'})
# To get only US total result

usdf = df[df['State'] == 'United States']
usdf
# cancer death rate

candf = df[df['Cause Name'] == 'Cancer']
candf.head()
#1 average death counts for all causes from 1999-2017

plt.style.use('ggplot')

avg_deaths = usdf.groupby('Year')['Deaths'].mean().sort_values(by = ['year'], ascending=False) #I don't get why it's not working
death_line = avg_deaths.plot(kind = 'line', figsize=(20,10), legend = True, linewidth=10)

# plot the line graph  
plt.title('Average death counts for all causes from 1999-2017',fontsize = 25, fontweight='bold')
plt.ylabel('Total death counts',fontsize = 18, fontweight='bold')
plt.xlabel('1999 to 2017', fontsize = 18, fontweight='bold')
plt.xticks(rotation=75)
plt.show()
#2 average death rate of cancer from 1999-2017

#style
plt.style.use('ggplot')

# average death rate - dereived from cancer only table
avg_Deathr = candf.groupby('State')['Death Rate'].mean().sort_values(ascending=False)
Deathr_bar = avg_Deathr.plot(kind = 'bar', figsize=(20,5), legend = True)

# plot the bar chart to show the average crude rate across states  
plt.title('Average death rate of cancer in each state per year',fontsize = 25, fontweight='bold')
plt.ylabel('Death rate',fontsize = 18, fontweight='bold')
plt.xlabel('States', fontsize = 18, fontweight='bold')
plt.xticks(rotation=75)
plt.show()

#3 The ratio of death counts of each cause in the United States (average number from 1999-2017)
# pie chart

# new style
plt.style.use('seaborn-dark')

# try to drop rows with value "all causes" under Cause Name column
# usdf = usdf[usdf.column[2] != 'All causes'] - not working...
usdf = usdf.drop(usdf[usdf['Cause Name']=='All causes'].index)

# get the final value (?)
avg_usdc = usdf.groupby('Cause Name')['Deaths'].mean().sort_values(ascending=False)

# pie plot
plt.title('The ratio of death counts of each cause in the United States',fontsize = 18, fontweight='bold')
avg_usdc.plot(kind='pie', y = 'Deaths', startangle=180, shadow=False, legend = False, fontsize=10, figsize=(30,5))


## I couldn't figure out how to avoid texts overlapping.