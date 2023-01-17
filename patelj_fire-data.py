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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import chardet

with open('../input/forest-fires-in-brazil/amazon.csv', 'rb') as rawdata:

    result = chardet.detect(rawdata.read(10000))

print(result)
data_path = '../input/forest-fires-in-brazil/amazon.csv'

df = pd.read_csv(data_path,encoding =  'ISO-8859-1',parse_dates = True)
df.describe()
df.head()
df.head()

year_mo_state = df.groupby(by = ['year','state', 'month']).sum().reset_index()

plt.figure(figsize=(12,4))

sns.lineplot(x='year',y='number',data=year_mo_state,estimator='sum',lw=3,err_style = None).set_title("Year vs Number of Fires in Brazil")

plt.figure(figsize=(12,4))

sns.lineplot(x='month',y='number',data=year_mo_state,estimator='sum',lw=3,err_style = None).set_title("The Frequency of Wildfires by Month")
plt.figure(figsize=(30,8))

sns.barplot(x='state',y='number',data=year_mo_state,lw=3).set_title("The Frequency of Wildfires in Different States")