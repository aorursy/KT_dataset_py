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
import seaborn as sns

from matplotlib import pyplot as plt
data=pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

data.head()
data.sample(10)   #Randomly selected 10 rows
data.info() #Properties of Data
data.describe()
data.isnull()
data.isnull().any()
data.isnull().sum()
data['children'].fillna(0,inplace=True)
data['children'].isnull().sum() 
data['country'].ffill(axis=0,inplace=True)
data['country'].isnull().sum()
data['agent'].fillna(data['agent'].mean(),inplace=True)
data['agent'].isnull().sum()
len(data['hotel'].unique())
len(data['customer_type'].unique()) 
plt.figure(figsize=(15,5))

sns.countplot(data=data[['hotel','arrival_date_month']],x='arrival_date_month',hue='hotel')