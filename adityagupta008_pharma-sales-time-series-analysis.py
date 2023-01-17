import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reading the data

hourly = pd.read_csv('/kaggle/input/pharma-sales-data/saleshourly.csv')

daily = pd.read_csv('/kaggle/input/pharma-sales-data/salesdaily.csv')

weekly = pd.read_csv('/kaggle/input/pharma-sales-data/salesweekly.csv')

monthly = pd.read_csv('/kaggle/input/pharma-sales-data/salesmonthly.csv')
#function to print shape of a given data

def print_shape(data):

    print('Rows : ',data.shape[0])

    print('Columns : ',data.shape[1])
print_shape(hourly)

print_shape(daily)

print_shape(weekly)

print_shape(monthly)
hourly.head(2)
daily.head(2)
weekly.head(2)
monthly.head(2)
#copy the data

hourly_original = hourly.copy()

daily_original = daily.copy()

weekly_original = weekly.copy()

monthly_original = monthly.copy()
#converting datatype of dates from object to Datetime

monthly['datum'] = pd.to_datetime(monthly['datum'], format= '%Y-%m-%d')

weekly['datum'] = pd.to_datetime(weekly['datum'], format= '%m/%d/%Y')

daily['datum'] = pd.to_datetime(daily['datum'], format= '%m/%d/%Y')

hourly['datum'] = pd.to_datetime(hourly['datum'], format= '%m/%d/%Y %H:%M')
#import datetime for dates and time realted calculations

import datetime as dt
#extracting year from dates

monthly['year'] = monthly['datum'].dt.year
#extracting month from dates

monthly['month'] = monthly['datum'].dt.month
#extracting day from dates

monthly['day'] = monthly['datum'].dt.day
#set index equal to the dates which will help us in visualising the time series

monthly.set_index(monthly['datum'], inplace= True)
monthly.head(2)
#define a function to plot yearly sales of every category of drug.

def plot_yearly_sales(column):

    monthly.groupby('year')[column].mean().plot.bar()#calculating yearly sales using groupby

    plt.title(f'Yearly sales of {column}')

    plt.xlabel('Year')

    plt.ylabel('Sales')

    plt.show()
#plotting yearly sales of each drug category

for i in monthly.columns[1:9]:#drug categories are from 1 to 8 index

    plot_yearly_sales(i) 
#lets see some statistics related to the data

monthly.describe()
#plot line curve to analyse monthly sales

def plot_line_curve(series):

    plt.figure(figsize= (15,5))

    series.plot(kind= 'line')

    plt.title(f'Monthly Sales of Drug : {col}')

    plt.show()
for col in monthly.columns[1:9]:

    plot_line_curve(monthly[col])
daily.columns
#extracting days from date

daily['day'] = daily['datum'].dt.day
#set dates as index

daily.set_index(daily['datum'], inplace= True)
#looking at sales data from 1st Jan, 2017 to 1st Feb, 2019

for col in daily.columns[1:9]:

    plot_line_curve(daily[col].loc['1/1/2017':'2/1/2017'])
#calculating total sales

monthly['total_sales'] = monthly['M01AB']

for cols in monthly.columns[2:9]:

    monthly['total_sales'] = monthly['total_sales']+monthly[cols]
monthly.groupby('month')['total_sales'].plot.bar(rot=45)

plt.xlabel('Date Time')

plt.ylabel('Total Sales')

plt.title('Total Sales of Drugs')

plt.show()