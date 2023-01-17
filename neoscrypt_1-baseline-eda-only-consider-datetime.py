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
baseline=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
baseline.head()
dt_idx = pd.DatetimeIndex(baseline['datetime'])
baseline['year'] = dt_idx.year
baseline['month'] = dt_idx.month
baseline['day'] = dt_idx.day
baseline['dayofweek'] = dt_idx.dayofweek
baseline['hour'] = dt_idx.hour
baseline['minute'] = dt_idx.minute
baseline['second'] = dt_idx.second
print(baseline.shape)
baseline.head()
baseline['weekend'] = 0
baseline.loc[baseline['dayofweek'] >= 5, 'weekend'] = 1

baseline.head()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18, 8)

sns.barplot(data=baseline, x="year", y="count", ax=ax1)
sns.barplot(data=baseline, x="month", y="count", ax=ax2)
sns.barplot(data=baseline, x="day", y="count", ax=ax3)
sns.barplot(data=baseline, x="hour", y="count", ax=ax4)
sns.barplot(data=baseline, x="minute", y="count", ax=ax5)
sns.barplot(data=baseline, x="second", y="count", ax=ax6)
baseline["year_month"] = baseline["year"].astype('str') + '-' + baseline["month"].astype('str')
baseline.head()
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18, 4)

sns.barplot(data=baseline, x="year", y="count", ax=ax1)
sns.barplot(data=baseline, x="month", y="count", ax=ax2)

figure, ax3 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.barplot(data=baseline, x="year_month", y="count", ax=ax3)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.pointplot(data=baseline, x="hour", y="count", hue="dayofweek", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.pointplot(data=baseline, x="hour", y="count", hue="weekend", ax=ax2)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = baseline.loc[baseline['weekend'] == 0]

sns.pointplot(data=target, x="hour", y="count", hue="year", ax=ax1)
target = baseline.loc[baseline['weekend'] == 0]

figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.pointplot(data=target, x="hour", y="count", hue="season", ax=ax1)
baseline = baseline.drop(columns=['minute', 'second'])
baseline.to_csv('eda_df.csv', index=False)
















































