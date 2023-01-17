# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv", parse_dates=['Date'])
train.head()
train.dtypes
min_date = train["Date"].min()
min_date
train["day_num"] = (train["Date"] - min_date).dt.days
train.head()
train.describe()
print("NA per Column:")
display(train.isna().sum())
train['geo_id'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)
train.head()
# plot ConfirmedCases
fig, ax = plt.subplots()
train.sort_values(by="Date").groupby('geo_id').plot.line(x='Date', y='ConfirmedCases', ax=ax, legend=False)
# plot Fatalities
fig, ax = plt.subplots()
train.sort_values(by="Date").groupby('geo_id').plot.line(x='Date', y='Fatalities', ax=ax, legend=False)
train.hist()
train.corr()