# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
d = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
d.head()
d.info()
d = d.dropna(subset = ['user_location'])
d.info()
d['date'] = pd.to_datetime(d['date'])
print(min(d['date']))
print(max(d['date']))
djuly = d['user_location'][(d['date'] < '2020-08-01 00:00:00')].value_counts()
daugust = d['user_location'][(d['date'] > '2020-08-01 00:00:00')].value_counts()
print(djuly[0:20])
print(daugust[0:20])
fig, axs = plt.subplots(figsize=(12, 7))
djuly[0:20].plot(kind = 'bar')


fig, axs = plt.subplots(figsize=(12, 7))
daugust[0:20].plot(kind = 'bar')