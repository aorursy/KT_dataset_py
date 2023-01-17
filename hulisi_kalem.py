# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Get first 10000 rows and print some info about columns
train = pd.read_csv("../input/train.csv", parse_dates=['srch_ci', 'srch_co'], nrows=10000)
train.info()
import seaborn as sns
import matplotlib.pyplot as plt
# preferred continent destinations
sns.countplot(y='hotel_continent', data=train)
# most of people booking are from continent 3 I guess is one of the rich continent?
sns.countplot(y='posa_continent', data=train)
# putting the two above together
sns.countplot(y='hotel_continent', hue='posa_continent', data=train)
# how many people by continent are booking from mobile
sns.countplot(y='posa_continent', hue='is_mobile', data = train)
# Difference between user and destination country
sns.distplot(train['user_location_country'], label="User country")
sns.distplot(train['hotel_country'], label="Hotel country")
plt.legend()
import numpy as np
# get number of booked nights as difference between check in and check out
hotel_nights = train['srch_co'] - train['srch_ci'] 
hotel_nights = (hotel_nights / np.timedelta64(1, 'D')).astype(float) # convert to float to avoid NA problems
train['hotel_nights'] = hotel_nights
plt.figure(figsize=(11, 9))
sns.boxplot(x='hotel_nights', y='hotel_country', data=train)
plt.figure(figsize=(11, 9))
sns.countplot(x="hotel_nights", data=train)
# distribution of the total number of people per cluster
src_total_cnt = train.srch_adults_cnt + train.srch_children_cnt
train['src_total_cnt'] = src_total_cnt
ax = sns.kdeplot(train['hotel_cluster'], train['src_total_cnt'], cmap="Purples_d")
lim = ax.set(ylim=(0.5, 4.5))