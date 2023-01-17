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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("../input/portblair/portBlair.csv")
df.info()
df.head()
df.describe()
# Highest precipitation in portblair

rank = df.sort_values('prcp', ascending=False).head(10).reset_index(drop=True)
rank.index = rank.index + 1
rank
plt.style.use('bmh')
df[['temp']].plot(subplots=True, figsize=(18,12));
condition = np.logical_and(df['year'].isin([2010,2019]), df['mo'] <= 11)
data = df[condition]

plt.subplots(figsize=(12,6))

sns.boxplot(x='mo', y='temp', hue='year', data=data)
plt.xlabel('Month')
plt.ylabel('Temerature (C)')
plt.title('Temperature Comparison');
plt.subplots(figsize=(12,6))
sns.barplot(x='mo',y='prcp', hue='year', data=data, ci=None)
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.title('Precipitation Comparison');


