# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/restaurant-order-data-for-a-month/Data.csv')
data.head()
data.tail()
data.info()
data.isna().sum()
data.dropna(inplace= True)
data.isna().sum()
data.columns
cols = ['TAKEAWAY/SEATING/PICKUP', 'SERVER','ITEM', 'QUANTITY', 'ALLERGIES', 'FOOD TYPE']
# check for the columns which has below 5 classes 
for column in cols:
    if len(data[column].unique()) <= 12:
        print("{}: {}".format(column ,data[column].unique()))
data['SERVER'].nunique()
data['ITEM'].nunique()
data.columns
# which arrangement is benifit
plt.figure(figsize= (15,5))
plt.style.use('seaborn')
sns.countplot(x = 'TAKEAWAY/SEATING/PICKUP', data = data)
plt.show()
data.groupby(['TAKEAWAY/SEATING/PICKUP'])['TOTAL'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
plt.show()
# Profitable revenue by Item (top 10)
data.groupby(['ITEM'])['TOTAL'].sum().sort_values(ascending = False)[:10].plot(kind = 'bar', figsize = (15,5))
plt.show()
# Profitable revenue by Foodtype (top 10)
data.groupby(['FOOD TYPE'])['TOTAL'].sum().sort_values(ascending = False)[:10].plot(kind = 'bar', figsize = (15,5))
plt.show()
# profitable week
data.groupby(['WEEK NUMBER'])['TOTAL'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
plt.show()
# from above bar graph we know week-3 has most profitable so, find which date and day
data[data['WEEK NUMBER'] == 'WEEK 3'].groupby(['WEEK NUMBER','DATE', 'DAY'])['TOTAL'].sum().sort_values(ascending = False)
data[data['WEEK NUMBER'] == 'WEEK 3'].groupby(['WEEK NUMBER','DATE', 'DAY', 'TAKEAWAY/SEATING/PICKUP'])['TOTAL'].sum().sort_values(ascending = False)
# profitable day in week 3
data[data['WEEK NUMBER'] == 'WEEK 3'].groupby(['WEEK NUMBER','DAY'])['TOTAL'].sum().sort_values(ascending = False)
weeks = ['WEEK 1', 'WEEK 2','WEEK 3','WEEK 4','WEEK 5']
for week in weeks:
    print(data[data['WEEK NUMBER'] == week].groupby(['WEEK NUMBER', 'DAY'])['TOTAL'].sum().sort_values(ascending = False))
weeks = ['WEEK 1', 'WEEK 2','WEEK 3','WEEK 4','WEEK 5']
for week in weeks:
    data[data['WEEK NUMBER'] == week].groupby(['WEEK NUMBER', 'DAY'])['TOTAL'].sum().sort_values(ascending = False).plot(kind = 'bar')
    plt.show()
# Profitable day
data.groupby(['DAY'])['TOTAL'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
plt.show()
data.columns
# profitable date
data.groupby(['DATE'])['TOTAL'].sum().plot(kind = 'line', figsize = (15,5))
plt.show()
data.groupby(['WEEK NUMBER', 'DATE', 'DAY'])['TOTAL'].sum().sort_values(ascending = False)
# most engaged server
data.groupby(['SERVER'])['TOTAL'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,5))
plt.show()
print(data['DATE'].max())
print(data['DATE'].min())
data.groupby(['Order_ID'])['TOTAL'].sum().sort_values(ascending = False)[:10]
# busiest time in a month..
data['TIME'].value_counts().plot(kind = 'line', figsize = (15,5))
plt.show()
# most valuable time
data.groupby('TIME')['TOTAL'].sum().sort_values(ascending = False).plot(kind = 'line', figsize = (15,5))
plt.show()
