# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

import calendar





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
la = pd.read_csv('../input/traffic-collision-data-from-2010-to-present.csv',parse_dates=['Date Reported'])



la.info()
dataframe_rows, dataframe_cols = la.shape

print("The dimensions of the Los Angeles Collission dataset is",dataframe_rows,"rows and",dataframe_cols,"columns.")
la.drop_duplicates()

la.info()
la.drop(["DR Number", # 

         "Area ID", # Area Name is better

         "Crime Code",

         "Premise Code",  # Premise Description is better

         "Date Reported", # Date Occurred is better 

         "Census Tracts", # not used

         "Council Districts", # not used

         "MO Codes", # not used

         "LA Specific Plans"], # not used

         axis=1, inplace=True)


la.describe()
la['Date'] =  pd.to_datetime(la['Date Occurred'], format='%Y-%m-%dT%H:%M:%S')

                             

#la['year_of_date'] = la['Date'].dt.year

#la['year'] = la['Date'].pd.year

#la['month'] = la['Date'].pd.month

la['year'] = pd.DatetimeIndex(la['Date']).year

la['month'] = pd.DatetimeIndex(la['Date']).month

la['weekday']= pd.DatetimeIndex(la['Date']).dayofweek
la_year = la[la['year']<2019] # Removing 2019 for being incomplete

fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(x = 'year',data = la_year,palette='bright')

ax.set_title('Collissions by year in LA');
la_2015 = la[la['year']==2015]



la_2016 = la[la['year'] == 2016]

la_2016.head()

la_2016.info()
# First let's visualize the number of crimes according to area name

fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(y="Area Name",data=la_2016,palette='bright')

ax.set_title('Collission in LA in 2016 by Area');
la_2016['Zip Codes'].describe()
fig,ax = plt.subplots(figsize=(12,8))

descending_order = la_2016['Area Name'].value_counts().sort_values(ascending=False).index

sns.countplot(data=la_2016,y="Area Name",order=descending_order)

ax.set_title('Collission in LA in 2016 by Area');
# First let's visualize the number of crimes according to area name

fig,ax = plt.subplots(figsize=(25,10))

sns.countplot(x="Zip Codes",data=la_2016,palette='bright')

ax.set_title('Collission in LA in 2016 by Zip Code');
fig,ax = plt.subplots(figsize=(12,8))

descending_order = la_2016['Victim Descent'].value_counts().sort_values(ascending=False).index

sns.countplot(data=la_2016,y="Victim Descent",order=descending_order)

ax.set_title('Victim Ethnic in collissions in LA in 2016 by Area');
la_2016['Victim Age'].describe()


fig,ax = plt.subplots(figsize=(12,18))

sns.countplot(y="Victim Age",data=la_2016,palette='bright')

ax.set_title('Victim Ages in collission in LA in 2016 ');
fig,ax = plt.subplots(figsize=(12,18))

descending_order = la_2016['Victim Age'].value_counts().sort_values(ascending=False).index

sns.countplot(data=la_2016,y="Victim Age",order=descending_order)

ax.set_title('Victim Age in collissions in LA in 2016 _ sorted ny number of collissions');
fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(data=la_2016,x="Victim Sex",palette='bright')

ax.set_title('Victim gender in collissions in LA in 2016');


fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(x = 'month',data = la_2016,palette='bright')

ax.set_xticklabels(["January", "February", "March", "April", "May", "June", "July","August","September","October","November","December"], fontsize=9)

ax.set_title('Collissions by months in LA in 2016');
fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(x = 'weekday',data = la_2016,palette='bright')

ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fontsize=10)

ax.set_title('Collissions by day of the week in LA in 2016');
fig,ax = plt.subplots(figsize=(34,18))

sns.countplot(x = 'Date',data = la_2016)



ax.set_title('Collissions by dates in LA in 2016');