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
from math import *
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
bookings = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
bookings.head()
bookings.tail()
bookings.info()
bookings.columns
bookings.describe()
corr = bookings[
    ['is_repeated_guest', 'previous_cancellations', 
     'previous_bookings_not_canceled', 'booking_changes', 
     'days_in_waiting_list', 'lead_time', 'adults', 
     'children', 'babies','is_canceled']
]

with sns.axes_style("white"):
    table = corr.corr()
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(18,7))
    sns.heatmap(table, cmap='Reds', mask=mask, vmax=.3, linewidths=0.5, annot=True,annot_kws={"size": 15})
month_sorted = ['January','February','March','April','May','June','July','August','September','October','November','December']
plt.figure(figsize=(14,6))
sns.countplot(bookings['arrival_date_month'], palette='pastel', order = month_sorted)
plt.xticks(rotation = 90)
plt.show()
bookings.drop(['agent', 'company', 'arrival_date_week_number'], axis=1, inplace=True)
# Lets look into the numbers of children accompanying the adults since there are few missing values in children column
bookings.children.value_counts()
plt.style.use('fivethirtyeight')

plt.figure(figsize=(14,6))
sns.countplot(x='hotel',data=bookings,hue='is_canceled',palette='pastel')
plt.show()
plt.figure(figsize=(14,6))
sns.countplot(x='deposit_type',data=bookings,hue='is_canceled',palette='pastel')
plt.show()

