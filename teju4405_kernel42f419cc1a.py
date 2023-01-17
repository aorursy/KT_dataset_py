# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df.head()
x=df.groupby('hotel')['hotel','lead_time','adults'].mean()

x.plot(figsize=(5,5))
x.plot.bar(figsize=(5,5))
df.describe()
x.plot.kde()
df['lead_time'].value_counts()
df.info()
import seaborn as sns

sns.regplot(df['lead_time'],df['required_car_parking_spaces'])
df['reserved_room_type'].value_counts()
df['previous_cancellations'].sum()
y=df.groupby('country')['country','adults','meal'].mean()

y.plot()
y.plot.pie(autopct='%1.1f%%',figsize=(50,40),startangle=140,subplots=True)
d=df['distribution_channel']

d
df['total_of_special_requests']
sns.regplot(df['stays_in_week_nights'],df['stays_in_weekend_nights'])
x.plot.pie(subplots=True)
x
df.tail()
df.to_csv('hotel_booking.csv')