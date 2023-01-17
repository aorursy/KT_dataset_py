# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))
bar = pd.read_csv("../input/bar_locations.csv")

party = pd.read_csv("../input/party_in_nyc.csv")

test_parties = pd.read_csv("../input/test_parties.csv")

train_parties = pd.read_csv("../input/train_parties.csv")
bar.info()
party.info()
test_parties.info()
train_parties.info()
ax1 = party.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.2,color = '#89C4F4', label = 'party')

ax1.set_facecolor('#F2F1EF')

ax2 = bar.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.2,color = '#013243', ax=ax1, label = 'bar')

p = plt.legend()

t = ax2.set_title("Geolocalization of parties and bars")
dates_created = pd.to_datetime([date for date in party["Created Date"]])



hours = dates_created.hour



total_call_by_hours = np.unique(hours.values, return_counts=True)



df = pd.DataFrame(total_call_by_hours[0], columns=['hour'])

df['calls'] = total_call_by_hours[1]



bp = sns.barplot(x="hour", y="calls", data=df)

bp = bp.set_title("Total police calls per hour in 2016 in NYC")
day_31_dec_2015_midday = pd.Timestamp("2015-12-31 12:00:00")

day_1_jan_2016_midday = pd.Timestamp("2016-01-01 12:00:00")



day_31_dec_2016_midday = pd.Timestamp("2016-12-31 12:00:00")

day_1_jan_2017_midday = pd.Timestamp("2017-01-01 12:00:00")



data = pd.DataFrame(party)

data["Created Date"] = pd.to_datetime(data["Created Date"])

data["Closed Date"] = pd.to_datetime(data["Closed Date"])

data["Delta Date"] = data["Closed Date"] - data["Created Date"]

data["Delta Date"] 



data1 = data[(data["Created Date"] > day_31_dec_2015_midday) & (data["Created Date"] < day_1_jan_2016_midday)]

data2 = data[(data["Created Date"] > day_31_dec_2016_midday) & (data["Created Date"] < day_1_jan_2017_midday)]
deltaMinutes_2015_2016 = data1["Delta Date"].astype('timedelta64[m]')
deltaMinutes_2015_2016 = pd.DataFrame(deltaMinutes_2015_2016)

deltaMinutes_2015_2016.describe()

g = sns.distplot(deltaMinutes_2015_2016).set_title("Histogram of sample from 2015-2016")
deltaMinutes_2015_2016 = data2["Delta Date"].astype('timedelta64[m]')

sample2016_2017 = deltaMinutes_2015_2016.sample(frac=0.1, replace=True)
pd.DataFrame(sample2016_2017).describe()
stats.normaltest(sample2016_2017)
g = sns.distplot(sample2016_2017).set_title("Histogram of sample from 2016-2017")
stats.ttest_ind(deltaMinutes_2015_2016, sample2016_2017, equal_var=False)