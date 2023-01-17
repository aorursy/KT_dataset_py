# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import time

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")

df.head()
df.isnull().sum()
df_times = df[['ID','Start_Time','End_Time']]

df_times['Start_Time'] = pd.to_datetime(df_times['Start_Time'])

df_times['End_Time'] = pd.to_datetime(df_times['End_Time'])

df_times.head()

df_times.info()
hours = df_times.groupby(df_times['Start_Time'].map(lambda x: x.hour)).count()

hours
plt.figure(figsize = (15,5))

sns.lineplot(x=hours.index, y=hours['Start_Time'])

plt.show()