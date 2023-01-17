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
file_path = "../input/hotel-booking-demand/hotel_bookings.csv"
df = pd.read_csv(file_path)
df.sample(10)
df.shape
df.info()
df.describe()
df.isnull().sum()
df['children'].fillna(0,inplace=True)
df['children'].isnull().sum()
df['country'].ffill(axis = 0,inplace=True)
df['country'].isnull().sum()
df.agent=df.agent.fillna(df.agent.mean())
df['agent'].isnull().sum()
df['hotel'].unique()
df['hotel'].value_counts()
df['customer_type'].unique()
df['customer_type'].value_counts()
import seaborn as sns
df.corr()
sns.heatmap(df.corr(),annot=True)
sns.pairplot(df)
