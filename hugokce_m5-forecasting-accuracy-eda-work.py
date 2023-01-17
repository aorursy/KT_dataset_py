# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import datetime





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print('M5 Forecasting - Accuracy EDA Work is beginning')
import pandas as pd

cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sales_tv = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

sample_sub = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

sp = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
cal.columns
cal.shape
cal.info()
cal.head(20)
cal.tail(10)
cal.describe()
cal.columns[cal.isnull().any()]
cal.date.value_counts()
cal.date.unique()
cal.wm_yr_wk.value_counts()
cal.weekday.value_counts()
cal.wday.value_counts()
cal.month.value_counts().sort_values()
cal.d.value_counts()
cal.event_name_1.value_counts()
cal.event_type_1.value_counts()
cal.event_name_2.value_counts()
cal.event_type_2.value_counts()
cal.snap_CA.value_counts()
cal.snap_TX.value_counts()
cal.snap_WI.value_counts()
sales_tv.info()
sales_tv.columns[sales_tv.isnull().any()]
sales_tv.shape
sales_tv.columns
sales_tv.describe()
sales_tv.d_1.value_counts()
sample_sub.columns
sample_sub.shape
sample_sub.describe()
sample_sub.info()
sample_sub.columns[sample_sub.isnull().any()]
sp.info()
sp.columns[sp.isnull().any()]
sp.columns
sp.shape
sp.describe()
cal.columns[cal.isnull().any()]
print('You\'ve reached end of analysis. In a short time it will be developed.')