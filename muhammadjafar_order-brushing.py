# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

def diff_times_in_seconds(t1, t2):

    # caveat emptor - assumes t1 & t2 are python times, on the same day and t2 is after t1

    h1, m1, s1 = t1.hour, t1.minute, t1.second

    h2, m2, s2 = t2.hour, t2.minute, t2.second

    t1_secs = s1 + 60 * (m1 + 60*h1)

    t2_secs = s2 + 60 * (m2 + 60*h2)

    return( t2_secs - t1_secs)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
order = '/kaggle/input/order_brush_order.csv'

order_data = pd.read_csv(order, parse_dates=True)
order_data.head()
order_data.info()
order_data['Dates'] = pd.to_datetime(order_data['event_time']).dt.date

order_data['Time'] = pd.to_datetime(order_data['event_time']).dt.time

order_data_sorted = order_data.sort_values(["Dates","Time"], ascending = True).reset_index(drop=True)

print(order_data_sorted)
delta = diff_times_in_seconds((order_data_sorted.iloc[0]["Time"]), (order_data_sorted.iloc[4]["Time"]))

print(delta)



