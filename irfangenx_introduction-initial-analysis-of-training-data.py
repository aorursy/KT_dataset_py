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
#Import libs
import matplotlib.pyplot as plt
#load train.csv data
train = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv')
train.shape
train.head(10)
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['Price'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="Oil Price",
       title="Oil Price: 31-12-2019 to 31-03-2020 ")
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_total_cases'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_total_cases",
       title="World_total_cases: 31-12-2019 to 31-03-2020 ")
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_new_cases'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_new_cases",
       title="World_new_cases: 31-12-2019 to 31-03-2020 ")
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_total_deaths'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_total_deaths",
       title="World_total_deaths: 31-12-2019 to 31-03-2020 ")
fig, ax = plt.subplots(figsize=(15,10))
ax.plot(train['Date'],train['World_new_deaths'])
plt.grid()
plt.xticks(rotation='vertical')
ax.set(xlabel="Date",
       ylabel="World_new_deaths",
       title="World_new_deaths: 31-12-2019 to 31-03-2020 ")