# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df.sample(10)
df.shape
df.info()
df.describe()
df.isnull().sum()
df['children'].unique()
#We can see that there exist nan value in children column
print("Null value in children column are",df['children'].isnull().sum())
# Filling nan value with 0
df['children'].fillna(0,inplace=True)
print("Null value in children column after using fillna",df['children'].isnull().sum())
df['country'].unique()
#We can see that there exist nan value in country column
print("Null value in country column are",df['country'].isnull().sum())
#Filling nan value with the last value of column.
df['country'].ffill(axis = 0,inplace=True)
print("Null value in country column after using fillna",df['country'].isnull().sum())
mean=np.mean(df['agent'])
df['agent'].unique()
#We can see that there exist nan value in agent column
print("Null value in agent column are",df['agent'].isnull().sum())
#Filling nan value of agent column with the mean of same column
df['agent'].fillna(mean,inplace=True)
print("Null value in agent column after using fillna",df['agent'].isnull().sum())
#Checking number of hotels and there types in the datset.
df['hotel'].value_counts()
#Checking number of customers and there types in the datset.
df['customer_type'].value_counts()
#Canceled=1, Not canceled= 0
canceled_data = df['is_canceled']
sns.countplot(canceled_data, palette='husl')
plt.show()

# Lets see in which type of hotel most booking happen and cancel
sns.countplot(x='hotel',hue="is_canceled", data=df,palette='husl')
plt.title("Cancelation rates in City hotel VS Resort hotel", size=18)
plt.show()

#Checking at which month average daily rate is high for both hotel types
plt.figure(figsize=(12,6))
sns.lineplot(x='arrival_date_month', y='adr', hue='hotel', data= df)
plt.show()
