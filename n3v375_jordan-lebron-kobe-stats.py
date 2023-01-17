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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('../input/michael-jordan-kobe-bryant-and-lebron-james-stats/allgames_stats.csv')
df.head()
kobe = df.loc[df['Player'] == 'Kobe Bryant']

kobe
jordan = df.loc[df['Player'] == 'Michael Jordan']

jordan
lebron = df.loc[df['Player'] == 'Lebron James']

lebron
print(f"--- Kobe ---\nMin: {kobe['PTS'].min()} \nMax: {kobe['PTS'].max()} \nAvg: {round(kobe['PTS'].mean())}")
print(f"--- Jordan ---\nMin: {jordan['PTS'].min()} \nMax: {jordan['PTS'].max()} \nAvg: {round(jordan['PTS'].mean())}")
print(f"--- Lebron ---\nMin: {lebron['PTS'].min()} \nMax: {lebron['PTS'].max()} \nAvg: {round(lebron['PTS'].mean())}")
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



kobe['Date'] = pd.to_datetime(kobe['Date'])

sns.boxplot(kobe['Date'].dt.year, kobe['PTS'], color='purple').set_title('Kobe')
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



jordan['Date'] = pd.to_datetime(jordan['Date'])

sns.boxplot(jordan['Date'].dt.year, jordan['PTS'], color='red').set_title('Jordan')
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



lebron['Date'] = pd.to_datetime(lebron['Date'])

sns.boxplot(lebron['Date'].dt.year, lebron['PTS'], color='yellow').set_title('Lebron')
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



sns.boxplot(kobe['Date'].dt.year, kobe['PTS'], color='purple').set_title('Kobe')

sns.boxplot(jordan['Date'].dt.year, jordan['PTS'], color='red').set_title('Jordan')

sns.boxplot(lebron['Date'].dt.year, lebron['PTS'], color='yellow').set_title('Jordan = Red / Kobe = Purple / Lebron = Yellow')