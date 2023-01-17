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
# import the library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#import the dataset

df=pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df.info()
# find no of rows and columns in the Dataset

nrow,ncol=df.shape

print(f"There are {nrow} rows and {ncol} columns in the Dataset")
# find null values in every columns

df.isnull().sum()
# check one condition

df.loc[(df["arrival_date_year"]==2015) & (df["arrival_date_month"]=="December")]
# find every year hotel booking

df.arrival_date_year.value_counts()
# find month wise columns booking

df.arrival_date_month.value_counts()
# Display null values in the columns

fig,axes = plt.subplots(1,1,figsize=(15,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()

# Here we can see that agent and company column in too Null values
sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Arrival_date_year",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Count Every Year Hotel Bookings",fontsize=27)

sns.countplot(x='arrival_date_year',data=df)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()

# check every year booking
# every month total booking in hotels

month_vise=df.arrival_date_month.value_counts().sort_values()

print(month_vise)

sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Hotel",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Booking in Resort Hotel Vs City Hotel", fontsize=27)

Booking=sns.countplot(x='arrival_date_month',data=df)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
# booking in Resort hotel vs city hotel

sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Hotel",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Booking in Resort Hotel Vs City Hotel", fontsize=27)

sns.countplot(x='hotel',data=df)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
# booking in city hotel and Resort hotel in seprate year 

sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Arrival_date_year",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Count in every year Booking in Resort Hotel Vs City Hotel", fontsize=27)

sns.countplot(x='arrival_date_year',data=df,hue='hotel')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
# check Booking is canceled or not

sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Arrival_date_year",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Check Booking is canceled or not", fontsize=27)

sns.countplot(x='arrival_date_year',data=df,hue='is_canceled')

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
# Check Booking deposite Status

sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Arrival_date_year",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Check Booking is Deposite or not",fontsize=27)



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.countplot(x='deposit_type',data=df)

plt.show()
# Check Top 10 country Hotel booking

sns.set_style("dark")

plt.figure(figsize=(13,6))

data1=df['country'].value_counts().sort_values(ascending=False).head(15)





Deposite=sns.barplot(x=data1.index,y=data1.values)

Deposite.set_xticklabels(labels=data1.index,fontsize=14)



plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel("Country",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Top 10 country Booking ",fontsize=27)

plt.show()
# show data customer type Wise

sns.set_style("dark")

plt.figure(figsize=(13,6))

plt.xlabel("Customer Type",fontsize=20)

plt.ylabel("Count",fontsize=20)

plt.title("Types of Customer",fontsize=27)

sns.countplot(x='customer_type',data=df)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()