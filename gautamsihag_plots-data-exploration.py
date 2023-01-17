# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000)
train = pd.read_csv("../input/train.csv", parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=100000)
train1["date_time"] = pd.to_datetime(train1["date_time"], format='%Y-%m-%d', errors="coerce")
train1["srch_ci"] = pd.to_datetime(train1["srch_ci"], format='%Y-%m-%d', errors="coerce")
train1["srch_co"] = pd.to_datetime(train1["srch_co"], format='%Y-%m-%d', errors="coerce")
train1["stay_span"] = (train1["srch_co"] - train1["srch_ci"]).astype('timedelta64[h]')
train1["search_span"] = (train1["srch_ci"] - train1["date_time"]).astype('timedelta64[D]')
train1["search_span"].head()
train1["stay_span"].describe()
train_bookings = train1[train1['is_booking'] == 1].drop('is_booking', axis=1)
train_clicks = train1[train1['is_booking'] == 0].drop('is_booking', axis=1)
train_clicks.is_mobile.head()
train_bookings_mobile = train_bookings[train_bookings['is_mobile'] == 1].drop('is_mobile', axis=1)
train_clicks_mobile = train_clicks[train_clicks['is_mobile'] == 1].drop('is_mobile', axis=1)
train_bookings_nonmobile = train_bookings[train_bookings['is_mobile'] == 0].drop('is_mobile', axis=1)
train_clicks_nonmobile = train_clicks[train_clicks['is_mobile'] == 0].drop('is_mobile', axis=1)
booked_hotels_country = train_clicks_nonmobile['stay_span']
booked_hotels_country_bookings, count = np.unique(booked_hotels_country, return_counts=True)
np.median(count)
count.max()
most_visited = booked_hotels_country_bookings[count >= 25]
most_visited
train_clicks_nonmobile.stay_span.head()
sns.countplot(y='stay_span', data=train_bookings_mobile)
sns.plt.title('stay_span wise destination distribution of 10m bookings')
plt.show()
sns.set(style="darkgrid")
ax = sns.countplot(x="stay_span", hue="is_mobile", data=train_bookings)
sns.set(style="darkgrid")
ax = sns.countplot(x="stay_span", hue="is_package", data=hotel_cluster_count)
