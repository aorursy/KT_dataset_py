# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#This is the orginal data



df = pd.read_csv('/kaggle/input/seattle-crime-data-to-present/Crime_Data.csv')

df


df = df[['Occurred Date','Reported Date','Crime Subcategory','Primary Offense Description','Precinct','Neighborhood']]

df['Reported Date'] = pd.to_datetime(df['Reported Date'])

df['Occurred Date'] = pd.to_datetime(df['Occurred Date'])

start_date = '01/01/2018'

end_date = '12/31/2018'

mask = (df['Reported Date'] > start_date) & (df['Reported Date'] <= end_date)

df_2018 = df.loc[mask]

df_2018
#reindex the 2018 data



df_2018 = df_2018.sort_values('Reported Date',ascending=True).reset_index(drop=True)

df_2018

plt.style.use('dark_background')
t = df_2018['Crime Subcategory'].value_counts().sort_values(ascending=True)



# get value of crime number in each crime type

temp = df_2018['Crime Subcategory'].unique()

crime_num = []

for i in temp:

    crime_num.append(t[i])

crime_num.sort()



#plot bar chart



cplot = t.plot(kind = 'barh', figsize=(20,10))

plt.title('Crime Ranking',fontsize = 20)

plt.ylabel('Type of Crimes',fontsize = 16)

plt.xlabel('Number of Crimes', fontsize = 16)



#  show value of each bar



for i, v in enumerate(crime_num):

    cplot.text(v + 3, i - 0.25, str(v), color='white', fontweight='bold')

plt.show()
t = df_2018['Neighborhood'].value_counts().sort_values(ascending=True)



# get value of crime number in each crime type

temp = df_2018['Neighborhood'].unique()

nbh_num = []

for i in temp:

    nbh_num.append(t[i])

nbh_num.sort()



#plot bar chart



nplot = t.plot(kind = 'barh', figsize=(20,20))

plt.title('Crime Frequency in Neighborhoods',fontsize = 20)

plt.xlabel('Number of Crimes',fontsize = 16)

plt.ylabel('Neighborhood Name', fontsize = 16)



#  show value of each bar



for i, v in enumerate(nbh_num):

    nplot.text(v + 3, i - 0.25, str(v), color='white', fontweight='bold')

plt.show()
# fliter the crimes only in sandpoint



df_sandpoint = df_2018[df_2018['Neighborhood'] == 'SANDPOINT'].reset_index(drop=True)

df_sandpoint

# Eliminate unnecessary columns 

# Sando

df_sp = df_sandpoint[['Reported Date','Crime Subcategory']]

df_sp
# Create a pie chart to show ratio of different types of crimes

df_sp['Crime Subcategory'].value_counts().plot(kind = 'pie',figsize = (10,10),autopct='%1.1f%%')
# Generate a table with months



df_2018['Month'] = pd.DatetimeIndex(df_2018['Reported Date']).month

df_2018
# crime_num_sp = df_sp['Crime Subcategory'].value_counts()

new_df_2018 = df_2018[['Month','Crime Subcategory']].groupby('Month').count()

new_df_2018

new_df_2018.plot(figsize=(15,6),legend = False)

plt.title('Crime Trends in Seattle',fontsize = 20)

plt.axis([1, 12, 0, 5000])

plt.xticks(np.arange(1,13,1)) #Reset the xticks label

plt.ylabel('Number of Crimes',fontsize = 18)

plt.xlabel('Month',fontsize = 18)

plt.show()
df_nbh_crime_rank = df_2018[['Crime Subcategory','Neighborhood']]

df_nbh_crime_rank['Count'] = 1

table = pd.pivot_table(df_nbh_crime_rank, values='Count', index=['Neighborhood', 'Crime Subcategory'],aggfunc=np.sum)



#Reorder the pivot table

new_table = table.reset_index().sort_values(['Neighborhood','Count'], ascending=[1,0]).set_index(['Neighborhood','Crime Subcategory'])

new_table
#Print out the first row in each neighborhood (which indicates the highest frequency crime)



nbh_name = df_2018['Neighborhood'].unique()

for i in nbh_name:

    print(new_table.loc[[i,'Crime Subcategory']].iloc[[0]])

    

#I don't know how to visualize this, I feel like looking at this table is clear enough
df_2018['Time Gap'] = (df_2018['Reported Date'] -df_2018['Occurred Date']).dt.days

tg = df_2018.sort_values(by='Time Gap',ascending = False)

#Average time gap

n_tg = tg[['Time Gap','Crime Subcategory']].groupby('Crime Subcategory').mean().sort_values(by='Time Gap',ascending = False)

print(n_tg)
n_tg.plot(kind = 'barh',figsize = (20,10))

plt.xlabel('Average Time Gap',fontsize = 18)

plt.ylabel('Crime Type',fontsize = 18)

plt.title('Time Gap Average in Different Crime Types')

plt.show()



#Rape has the longest time gap between 'Occurred Date' and 'Reported Date'
#Print out histagram of average time gap



n_tg.hist()