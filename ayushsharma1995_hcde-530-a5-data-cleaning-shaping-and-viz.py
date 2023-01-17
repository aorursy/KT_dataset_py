# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/WA_CRIME_DATA.csv')

df.head()
#Removing extraneous characters from column headers that might hamper reading the column headers in the data

df.columns = df.columns.str.replace('\n', ' ')

df.columns = df.columns.str.replace('- ', '')

df.columns = df.columns.str.replace('/', '')

df.columns = df.columns.str.replace('Population1', 'Population')



#Remove commas from number values to change their data type later

df = df.replace(',','', regex=True)



#Drop the column agency type since we are only concerned with agency/city names

del df['Agency Type']
df.head()
#Replace all the 'NaN' in the dataframe with zeroes.

df = df.fillna(0)

        

#Using built-in Pandas method to change the column datatype from object to int

df['Population'] = df['Population'].astype(str).astype(int)

df['Total Offenses'] = df['Total Offenses'].astype(str).astype(int)

df['Crimes Against Persons'] = df['Crimes Against Persons'].astype(str).astype(int)

df['Crimes Against Property'] = df['Crimes Against Property'].astype(str).astype(int)

df['Crimes Against Society'] = df['Crimes Against Society'].astype(str).astype(int)
#Seeing the data types of columns in our data frame

df.dtypes
#We use the built-in function 'n-largest' in pandas that grabs the n largest values in the column of a dataframe

most_pop_df = df.nlargest(6, 'Population')



city_labels = []

city_values = []



#Adding the most popular city names in a list that would be used as labels in visualization as well

for city in most_pop_df['Agency Name']:

    city_labels.append(city)



#Plotting the pie-chart for most populous cities and the their total offenses

most_pop_df.plot(kind='pie', y='Total Offenses', labels=city_labels, figsize=(14,7))



#What about the least populous cities and crime there?

# least_pop_df = df.nsmallest(6, 'Population')

# rare_city_labels = []

# for city in least_pop_df['Agency Name']:

#     rare_city_labels.append(city)

# least_pop_df.plot(kind='pie', y='Total Offenses', labels=rare_city_labels, figsize=(14,7))

#To create a histogram of cities with crime breakdown, we only need the specific columns to begin with

new_df = df[['Agency Name', 'Crimes Against Persons', 'Crimes Against Property', 'Crimes Against Society']].copy()



#Once we have the copy dataframe, we plot the crimes only for the populous cities

new_df[(new_df['Agency Name'].isin(city_labels))].plot.bar(x='Agency Name', figsize=(14,7))



#Creating a new dataframe that has all the Seattle related crime info

seattle_df = df[df['Agency Name'] == 'Seattle']



#Since we only care about specific crime types, dropping the initial generic columns 

seattle_df = seattle_df.drop(['Population', 'Total Offenses', 'Crimes Against Society', 'Crimes Against Persons', 'Crimes Against Property'], axis=1)



seattle_df
seattle_new = seattle_df.T
#Renaming the column header

seattle_new.columns = ['Frequency']



#Since we are dealing with Seattle only

seattle_new = seattle_new.drop(['Agency Name'])



#Changing the data type of the column

seattle_new['Frequency'] = seattle_new['Frequency'].astype(str).astype(float)

#Now plotting a horizontal bar graph of all the crimes in Seattle by frequency

seattle_new.plot.barh(figsize=(20,20))
#Another way to visualize the above data

seattle_new.plot.area(figsize=(20,10))