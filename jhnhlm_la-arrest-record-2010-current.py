import numpy as np

import pandas as pd

import warnings

import datetime as dt

import matplotlib.pyplot as plt

import geopandas as gpd





warnings.filterwarnings('ignore')



df = pd.read_csv(r'../input/dui.csv', sep = ';', index_col=0)

pd.set_option('display.max_columns', None) #to display all the columns



df.tail()





dui = df[df['Charge'] == '23152(A)VC']

dui.head()

dui.isnull().sum()

x = dui['Sex Code'].value_counts(normalize=True)

labels = ['Male','Female']

colors = ('#89cff0','#f4c2c2')

fig1, x1 = plt.subplots(figsize=(20,10))

plt.pie(x, labels = labels, autopct='%.1f%%', colors = colors, textprops={'fontsize': 18})

plt.axis('equal')

plt.suptitle('Arrested for DUI?', fontsize=30)

plt.show()
ages = dui.groupby('Age').size()

ages.plot.bar(figsize=(30,10))

plt.ylabel("Frequencies")

plt.title('LA  County Driving Under the Influence Age Data \n 2010 - 2019', fontdict = {'fontsize' : 35})
years = pd.to_datetime(dui['Arrest Date'])

byYears = years.groupby(years.dt.year).size()

byYears.plot(marker='o', markerfacecolor='blue', markersize=12, color ='skyblue', linewidth=4, figsize=(20,10))

plt.axis([2009, 2020, 0, 15000])

plt.ylabel('Frequencies')

plt.xticks([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])

plt.title('Los Angeles DUI Trends \n 2010 - 2019', fontdict = {'fontsize' : 35})

plt.show()


