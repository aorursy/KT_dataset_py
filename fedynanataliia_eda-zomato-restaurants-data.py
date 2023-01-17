# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); 
df = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")

code_of_country = pd.read_excel('../input/Country-Code.xlsx')

df1 = pd.merge(df, code_of_country, on='Country Code')

df1.head(10).T
df1['Country'].unique()
df1['Country'].value_counts().head(5)
df1['City'].unique()
df1['City'].value_counts().head(5)
df1.groupby('Cuisines')['Aggregate rating'].mean().plot.hist(orientation='vertical', color='pink', figsize=(5,5))

plt.figure()
df1.groupby('Aggregate rating')['Average Cost for two'].mean().plot(kind='bar', figsize=(10,10), color='blue')

plt.xlabel('Aggregate rating', color='g')

plt.ylabel('Average Cost for two', color='g')

plt.figure();
print('Has table booking = ', df1[df1['Has Table booking']=='Yes']['Aggregate rating'].mean())

print('Hasn`t table booking= ', df1[df1['Has Table booking']=='No']['Aggregate rating'].mean())
df1.groupby('Has Table booking')['Aggregate rating'].mean().plot(kind='bar', color='pink', figsize=(5,5))

plt.xlabel('Has Table Booking', color='grey')

plt.ylabel('Aggregate rating', color='grey') 

plt.figure();
df1.groupby('Rating text')['Longitude', 'Latitude'].mean().plot(kind='bar', figsize=(5,5))

plt.figure();
df1['Restaurant Name'].value_counts().head(5)
df1['Restaurant Name'].value_counts().head(5).plot(kind='bar', color='black', figsize=(10,10))

plt.xlabel('Name of restaurant', color='g')

plt.ylabel('Amount of each restaurant', color='g')

plt.figure()
df1['Cuisines'].value_counts()
df1['Cuisines'].value_counts().head(10).plot(kind='bar', figsize = (10,10), color='black')

plt.xlabel('Popularity of cuisines')

plt.figure()