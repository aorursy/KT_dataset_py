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
anzdata = pd.read_csv('../input/anz-csv/ANZ dataset.csv')
anzdata.head()
# check columns in dataframe
anzdata.info()
# mean amount of transactions executed per day
anzdata.groupby(['date']).sum()['amount'].mean()
# Convert the date column from string to datetime type
anzdata['date']=pd.to_datetime(anzdata['date'])

# Number of transactions made per month
anzdata.groupby(anzdata['date'].dt.strftime('%B'))['amount'].count()
# Get the average amount of transactions made by customers per month
anzdata.groupby(anzdata['date'].dt.strftime('%B'))['amount'].mean()
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
%matplotlib inline
#import folium
#from folium import Choropleth, Circle, Marker
#from folium.plugins import HeatMap, MarkerCluster
#import math
# The visualisation represents the amount of transaction done by females and males based on status. 
sns.set(font_scale=1.4)
anzdata['gender'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Gender", labelpad=14)
plt.ylabel("Count", labelpad=14)
plt.show()
# Creating Month by using the date gives so can be useful for EDA 
anzdata['month'] = anzdata['date'].dt.month_name()
anzdata['month'].head()
# Month where highest number of transaction took place
plt.figure(figsize=(18,8))
sns.countplot(x='month' , data=anzdata)
# Month where highest number of transaction took place based on gender
plt.figure(figsize=(18,8))
sns.countplot(x='month' ,hue='gender', data=anzdata)
plt.title('Month where highest number of\n'+'transaction took place based on gender',bbox={'facecolor':'0.9', 'pad':5})
# Average transactions made each month
month_grp = anzdata.groupby(['month_name'])
avg_month = month_grp['amount'].mean()
fig,ax = plt.subplots(figsize=(16,8)) # (height,width)
# my_colors = ['r','b','k','y','m','c','#16A085','salmon' , '#32e0c4'] #Colors for the bar of the graph
print(avg_month);
avg_month.plot.barh()
ax.set(
    title='Average transaction made my customer on average each month',
    xlabel='Average amount',
    ylabel='Month Name '
)
print(anzdata['merchant_state'].value_counts())
plt.figure(figsize=(10,7))
sns.countplot(anzdata['merchant_state'])
plt.title('Number of transaction\n' 'done in each state',bbox={'facecolor':'0.9', 'pad':5})
plt.show()

# Distribution of Age of the customers.
plt.figure(figsize=(10,7))
sns.distplot(anzdata['age']);
plt.title('Distribution of customers based on age group' , )
# Number of debit and credit transaction
plt.figure(figsize=(10,7))
print(anzdata['movement'].value_counts())
sns.countplot(anzdata['movement'])
# Top 10 customers 
top_cust = anzdata['first_name'].value_counts(sort=True).nlargest(10)
top_cust
# Least 10 Customers who made transaction
tail_cust = anzdata['first_name'].value_counts(sort=True).nsmallest(10)
tail_cust
