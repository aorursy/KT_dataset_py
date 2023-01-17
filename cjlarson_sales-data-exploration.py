
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # date time processing library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv",parse_dates=['date'])
df['date'][0].day
df.head()
# Any results you write to the current directory are saved as output.

#create a break down of day, month, and year from the date column using some hack and slash python string slinging

#trying to figure out how to break out pandas timestamp series into day, month, year etc while still being in the data frame
dates = df.date
dates
df['dayofmonth'] = dates.dt.day
df['month'] = df.date.dt.month
df['year'] = df.date.dt.year
df['weekday'] = df.date.dt.weekday
df.head()

#I think for the most part these are categorical variables the more i think about it, the only thing that is not categorical is the overall date from the beginning of the dataset 
#days since 2013-01-01

df['overalldays'] = df.date-df.date.min()
df['overalldays'] = df['overalldays'].dt.days
df.head()