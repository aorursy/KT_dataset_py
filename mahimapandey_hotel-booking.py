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
df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.sample(10)#Print random 10 rows of the dataset.
df.describe()#Show the properties of the dataset.
df.shape
df.isnull().sum()#Check the data for any null values, count the number of null values in each column.
df['children'].fillna(0)#Replace the null values of “children” column by 0.
df['country'].fillna(method='ffill')#Replace the null value of “country” column by the last value.
df['agent'].fillna(df['agent'].mean())#Replace the null values in the “agent” column by mean of that column.
df['hotel'].unique()#Check how many different “hotel” types are there.
df['customer_type'].unique()#Check how many different “customer” types are there.
import seaborn as sns
from matplotlib import pyplot as plt
df.columns
df.groupby([ 'arrival_date_month']).size().plot.bar(figsize=(12,5))
fig, ax = plt.subplots(figsize=(15,5))
sns.countplot(data = df[['hotel', 'arrival_date_month']], x = 'arrival_date_month', hue = 'hotel')