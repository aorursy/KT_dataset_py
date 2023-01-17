# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import Numpy
import numpy as np

# Import Pandas
import pandas as pd

# access data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# open the csv data
pod = pd.read_csv('../input/p-of-d-males-vs-femalespractice/prevalence-of-depression-males-vs-females.csv')
# read the csv and see what it's like
pod.head()
# get a general view of data
pod.shape
# Delete row when Prevalence in males or Prevalence in females is NaN
pod = pod.dropna(axis=0, how='any')
# check again see how many rows are dropped
pod.shape
# Same numbers? Why I failed??????????????????????????????????
# Anyway, move on. Get US data for info viz
uspod = pod[pod['Entity'] == 'United States']
uspod
# Import matplotlib, pandas datareader, and datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
# browse styles
plt.style.available
# Style the visuals
plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-poster')

# Histogram? Does not make sense here
hist = uspod.hist()
# Visualize the trend of US population with a line chart

# x = uspod[['Year']]
# y = uspod[['Population']]
uspod.plot(x='Year', y='Population')
# Visualize the trend of US Prevalence of depression of male/female
# x = uspod[['Year']]
# y1 = uspod[['Prevalence in males (%)']]
# y2 = uspod[['Prevalence in females (%)']]

uspod.plot(kind="scatter", x='Year', y='Prevalence in males (%)', c='purple')
uspod.plot(kind="scatter", x='Year', y='Prevalence in females (%)', c='orange')

# Why I can't put the two scatterplots in one visual???