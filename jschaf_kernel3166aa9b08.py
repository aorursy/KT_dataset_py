# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
marathon_data = pd.read_csv('../input/marathon_results_2017.csv')
marathon_data['Official Time'].describe()
males = marathon_data[marathon_data['M/F'] == 'M']

females = marathon_data[marathon_data['M/F'] == 'F']



def total_seconds(duration):

    h, m, s = duration.split(':')

    return int(h) * 3600 + int(m) * 60 + int(s)



def to_hms(seconds):

    s = int(seconds)

    h = s // 3600

    m = (s - h * 3600) // 60

    s = (s - h * 3600 - m * 60)

    return f'{h:02}:{m:02}:{s:02}'



def to_hm(seconds):

    return to_hms(seconds)[:-3]
bad = []

for i in range(20000):

    r = total_seconds(to_hms(i))

    if i != r:

        bad.append(i)

bad
male_times = males['Official Time'].apply(total_seconds).sort_values()

female_times = females['Official Time'].apply(total_seconds).sort_values()
male_times.describe()
percentiles = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

male_percentiles = male_times.quantile(percentiles)

female_percentiles = female_times.quantile(percentiles)
combined = pd.DataFrame({'male': male_percentiles, 'female': female_percentiles})



ax = combined.plot.bar()
all_percentiles = pd.concat([male_percentiles.apply(to_hm), female_percentiles.apply(to_hm)], axis=1)

all_percentiles.columns = ['Male', 'Female']

all_percentiles