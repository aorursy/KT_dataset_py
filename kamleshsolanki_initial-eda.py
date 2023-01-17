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
import pandas as pd

hotel_data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
hotel_data.head()
hotel_data.describe()
hotel_data.isnull().sum(axis=0)
hotel_data.info()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

corr = hotel_data.corr()

sns.heatmap(corr)
corr.head(len(corr))
hotel_data['hotel'].unique()
booking_cancel = hotel_data['hotel'].value_counts()

booking_cancel.plot(kind='bar')
booking_cancel = hotel_data['is_canceled'].value_counts()

booking_cancel.plot(kind='bar')
city_hotel = hotel_data[hotel_data.hotel == 'City Hotel']['is_canceled'].value_counts()

resort_hotel = hotel_data[hotel_data.hotel == 'Resort Hotel']['is_canceled'].value_counts()

df = pd.DataFrame({'City Hotel':city_hotel,'Resort Hotel':resort_hotel}).T

df.plot(kind = 'bar')
