# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt



df = pd.read_csv("../input/PAM.csv", parse_dates=['Date'])

df.head(10)
df.describe()
df = df.sort_values(by='Date')

df.set_index('Date',inplace=True)

ax = plt.pyplot.gca()

df['Close'].plot(figsize=(16, 12), ax=ax)

avg = df['Close'].rolling(200, win_type ='triang').mean() 

avg.plot(figsize=(16,12), ax = ax)
plt.pyplot.figure(figsize=(16,12))

plt.pyplot.scatter(df['Close']-df['Open'], np.log(df['Volume']), c=df['Close']-df['Open'])
