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
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
data1=pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

data1.sample(10) #random rows

data1.shape #properties and shape
data1.info()
data1.describe
data1.isnull().any() #check for null values
data1.isnull()
(data1.isnull().sum()).sum() #total number of null values in dataset
data1.isnull().sum() #number of null values in each column
bins=pd.cut(data1['arrival_date_week_number'],6,labels=["0-9 weeks","9-18 weeks","18-27 weeks","27-36 weeks","36-45 weeks","45-54 weeks"])

pd.value_counts(bins,sort=False) #binning the data according to number of bookings in 9 weeks

bins.dtype

plt.figure(figsize=(10,5))

pd.value_counts(bins,sort=False).plot(kind='bar',width=0.3)

plt.xlabel("weeks in a year") 

plt.ylabel("number of bookings") 

plt.title("Bookings done in diffeent week numbers of the year") 

plt.grid()

plt.show() 
#bins[bins.value_counts()].max()
print("most number of guests visited are",pd.value_counts(bins).max())
data1['children'].fillna(0,inplace=True) #replace null values of children with 0
data1['children'].isnull().any()
data1['country'].fillna(method='ffill',inplace=True) #replace null values of country by last value
data1['country'].isnull().any()
l1=((data1['agent']).mean(skipna=True))

data1['agent'].fillna(l1,inplace=True) #Replace the null values in the “agent” column by mean of that column.
data1['agent'].isnull().sum() #checking the number of null values present in agent column
no_hotel=data1['hotel'].unique()

len(no_hotel) #number of different hotel types
no_ct=data1['customer_type'].unique()

len(no_ct) ##number of different customer types
plt.figure(figsize=(15,5))

sns.countplot(x='arrival_date_month',hue='hotel',data=data1)