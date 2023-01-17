# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import Basic Libraries for Data Analysis



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Read Honey Production Dataset.



hp = pd.read_csv(r'/kaggle/input/honey-production/honeyproduction.csv')
#Reading First five Rows



hp.head() 
#Reading Last Five Rows



hp.tail()
#Summarizing hp dataset



hp.describe()
# Creating summary table to understand the trend using year variable



hp_year = hp[['numcol','totalprod','year','yieldpercol','stocks','prodvalue']].groupby('year').sum()



hp_year.head()
# Resetting index value



hp_year.reset_index(level=0,inplace=True)

hp_year.head()

# Visualizing the trend of Yield per Colony from year 1998 to 2012



plt.figure(figsize=(25,8))

plt.plot(hp_year['year'],hp_year['yieldpercol'])

plt.title('Trend of Honey Yield per Colony' ,fontsize=25)

plt.xlabel('Year',fontsize=25)

plt.ylabel('Yield per Colony',fontsize = 25)
# Visualizing the total honey production from the year 1998 to 2012.



plt.figure(figsize=(25,8))

plt.plot(hp_year['year'],hp_year['totalprod'])

plt.title('Total Honey Production in USA' ,fontsize=25)

plt.xlabel('Year',fontsize=25)

plt.ylabel('Total Production of Honey (lbs.)',fontsize = 25)
# Visualizing the trend of Production Value from year 1998 to 2012



plt.figure(figsize=(25,8))

plt.plot(hp_year['year'],hp_year['prodvalue'])

plt.title('Trend of Production Value' ,fontsize=25)

plt.xlabel('Year',fontsize=25)

plt.ylabel('Production Value',fontsize = 25)
# Group the dataset by states and using sum method to get the total honey production value descending order. 



US_state = hp[['state','totalprod','yieldpercol']].groupby('state').sum()

US_state.reset_index(level=0,inplace=True)

US_state.sort_values(by='totalprod',ascending=False,inplace=True)

US_state.head()
#Creating a Bar chart to visualize the total honey production by states.



plt.figure(figsize=(20,7))

sns.barplot(x=US_state['state'],y = US_state['totalprod'])

plt.title('Statewise Total Honey production in USA',fontsize =20)

plt.xlabel('States',fontsize=20)

plt.ylabel('Total Production of Honey in USA',fontsize=20)
# Creating a table to find out maximum production value from the states



US_state_max = hp[['state','totalprod']].groupby('state').max()

US_state_max.reset_index(level=0,inplace=True)

US_state_max.columns = ['State','Max Prod']

US_state_max.head()
# Creating a table to find out minimum production value from the states



US_state_min = hp[['state','totalprod']].groupby('state').min()

US_state_min.reset_index(level=0,inplace=True)

US_state_min.columns = ['State','Min Prod']

US_state_min.head()
# Merging the Max Prod and Min Prod varible to find the range.



st_range = pd.merge(US_state_max,US_state_min,how='inner',on='State')

st_range.head()
#Create a Per_Change Column in the st_range dataset to understand honey production changes by states.





st_range['Per_Change'] = ((st_range['Max Prod']-st_range['Min Prod'])/st_range['Max Prod'])*100

st_range.sort_values(by='Per_Change',ascending=False,inplace=True)

st_range.head()

#Create a Bar chart to visualize the statewise decline trend.



plt.figure(figsize=(20,7))

sns.barplot(x='State',y='Per_Change',data= st_range)

plt.title('Statewise Production Decline Trend',fontsize=20)

plt.xlabel("State",fontsize=15)

plt.ylabel("% Decline",fontsize=15)