# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')
'''
convert to datetime
'''
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
df['date'] = pd.to_datetime(df['Date'])
df['day'] = (df['date']).dt.day
df['month'] = (df['date']).dt.month
df['year'] = (df['date']).dt.year
df['dayofweek']= (df['date']).dt.dayofweek.map(dayOfWeek)

df['time']= pd.to_datetime(df['Time'])
df['hour']= (df['time']).dt.hour
'''all columns (features)'''
df.columns
#Data of Average quantity of product lines bought across days of the week
a=df.groupby(['Product line', 'dayofweek'])['Quantity'].mean().reset_index()
result= a.pivot(index='Product line', columns= 'dayofweek', values= 'Quantity')
print (result)  #6,31
#Heatmap of Average quantity of product lines bought across days of the week
fig, ax= plt.subplots(figsize= (31,6))
title= "Heatmap of sale per day"
plt.title(title, fontsize= 14)
tt1= ax.title
tt1.set_position([0.5, 1.05])
sns.heatmap(result, fmt="",cmap='jet', linewidths= 0.30, ax=ax)
#data of number of invoices generated per day of week by each gender
b= df.groupby(['Gender', 'dayofweek'])['Invoice ID'].count().reset_index()
pivot_b = b.pivot(index='dayofweek', columns='Gender', values='Invoice ID')
print (pivot_b)
#bar graph of number of invoices generated per day of week by each gender
pivot_b.plot.bar(stacked=False, figsize= (10,10))
#data of gender wise distribution of rating per product line
c=df.groupby(['Gender', 'Product line'])['Rating'].mean().reset_index()
pivot_c = c.pivot(columns='Gender', index='Product line', values='Rating')
print (pivot_c)
print (pivot_c.shape)
#line plot of gender wise distribution of rating per product line
lines = pivot_c.plot.line(rot=90, colormap='Spectral_r')
#data of hour wise buying per day of the week
d=df.groupby(['dayofweek', 'hour'])['Invoice ID'].count().reset_index()
pivot_d = d.pivot(columns='dayofweek', index='hour', values='Invoice ID')
print (pivot_d)
#stacked plot of hour wise buying per day of the week
pivot_d.plot.bar(stacked=True, figsize= (10,10)).legend(bbox_to_anchor=(1.4, 1))