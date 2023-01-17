# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from fbprophet import Prophet



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

data.head()
data.columns
data.hotel.value_counts()
hoteltype = data.groupby("hotel").is_canceled.count().reset_index()

hoteltype.columns = ["hotel","count"]

sns.set(style = "whitegrid")

ax = sns.barplot(x = "hotel", y = "count", data = hoteltype)
cancel = data.groupby(['hotel', 'is_canceled']).lead_time.count().reset_index()

cancel.columns = ['hotel', 'is_canceled', 'count']

ax = sns.barplot (x = "hotel", y = 'count', hue = 'is_canceled', data = cancel)
data["reservation_status_date"] = pd.to_datetime(data["reservation_status_date"])

is_cancelled_plot = data.groupby("reservation_status_date").is_canceled.sum().reset_index().sort_values(by = ["reservation_status_date"])

is_cancelled_plot.head()
is_cancelled_plot.dtypes
dims = (12, 9)

fig, ax = plt.subplots(figsize = dims)

sns.lineplot(ax=ax,x="reservation_status_date", y="is_canceled", data=is_cancelled_plot)
is_cancelled_plot.tail()
days =  pd.date_range('2015-01-01', '2017-09-14', freq = 'D')

is_cancelled_fill = pd.DataFrame({"reservation_status_date": days})

is_cancelled_fill = pd.merge(is_cancelled_plot[2:], is_cancelled_fill, on = 'reservation_status_date', how = 'outer')

is_cancelled_fill = is_cancelled_fill.fillna(0)

is_cancelled_fill.head()
dims = (12, 9)

fig, ax = plt.subplots(figsize = dims)

sns.lineplot(ax=ax,x="reservation_status_date", y="is_canceled", data=is_cancelled_fill)
AllindexOutlier = []

df_table = is_cancelled_fill['is_canceled'].copy()

Q1 = df_table.quantile(0.25)

Q3 = df_table.quantile(0.75)

IQR = Q3 -Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 - 1.5 * IQR

print('Lower bound is' + str(lower_bound))

print('Upper bouns is' + str(upper_bound))

print(Q1)

print(Q3)

outliers_vector = (df_table < (lower_bound)) | (df_table > (upper_bound) )

outliers_vector

outliers = df_table[outliers_vector]

listOut=outliers.index.to_list()

for t in listOut:

    AllindexOutlier.append(t)
AllindexOutlier[0:15]
for i in AllindexOutlier:

    # i filled with average

    is_cancelled_fill.loc[i,"is_canceled"]=is_cancelled_fill.is_canceled.mean()
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.lineplot(ax=ax,x="reservation_status_date", y="is_canceled", data=is_cancelled_fill)