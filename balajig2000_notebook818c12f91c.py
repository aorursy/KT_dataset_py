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
df1=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df1.sample(10)
df1.info()

df1.describe()
df1.isnull()
a=df1.isnull().sum()
a.sum()

df1.isnull().sum()
df1['children'].fillna(0)

df1['country'].ffill()
df1['agent'].fillna(df1['agent'].mean())

df1['hotel'].unique()

df1['customer_type'].unique()
import seaborn as sns
from matplotlib import pyplot as plt

labels = df1['hotel'].value_counts().index.tolist()
sizes = df1['hotel'].value_counts().tolist()
explode = (0, 0.1)
colors = ['lightgreen', 'coral']
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.show()

