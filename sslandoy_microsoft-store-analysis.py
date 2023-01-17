# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
%matplotlib inline
import plotly.graph_objs as go


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/windows-store/msft.csv')
df.info()
#Identify any null values, remove if there's any null values
df.isna().any() 
df.dropna(axis=0, inplace=True)
df.isna().any()

#Clean Price column to have uniform values
df['Price'] = df['Price'].replace('Free', 0)
df['Price'] =  df['Price'].replace('\₹', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'],errors='coerce')

df
free_app = round((df["Price"] == 0).sum() / len(df.index) * 100)
paid_app = round((df["Price"] != 0).sum() / len(df.index) * 100)

print("Microsoft Store consist of " + str(free_app) + "% free applications and " + str(paid_app) + " % paid applications." )

fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=12
labels = ['Free Application', 
         'Paid Applications']
percentages = [97, 3]
explode=(0.1,0)
ax.pie(percentages, explode=explode, labels=labels, autopct='%1.0f%%', 
       shadow=False, startangle=0,   
       pctdistance=1.2,labeldistance=1.4)
ax.axis('equal')
#ax.set_title("Distribution of Applications by Price")
ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8))


sum_price_cat = df.groupby(['Category'])['Price'].sum()
sum_price_cat.sort_values(ascending=False)
fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=12
labels = ['Business', 
         'Developer Tools',
         'Books']
percentages = [14606.4, 10042, 8564]
explode=(0.1,0, 0)
ax.pie(percentages, explode=explode, labels=labels, autopct='%1.0f%%', 
       shadow=False, startangle=0,   
       pctdistance=1.2,labeldistance=1.4)
ax.axis('equal')
#ax.set_title("Distribution of Applications by Category")
ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8))


# Percentage of free books category
new = df.loc[df["Price"] == 0.0]
total_free_app = len(new)
total_free_app_pct = (new["Category"].value_counts() / total_free_app) * 100
total_free_app_pct.sort_values(ascending=False)

fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=12
labels = ['Music',
          'Health & Fitness',
          'Lifestyle',
          'News & Weather',
          'Kids & Family',
          'Social',
          'Food & Dining',
          'Navigation & Maps',
          'Multimedia Design',
          'Government & Politics']
percentages = [14.5, 10.2, 9.5, 9.3, 6.6, 6.3, 3.2, 2.9, 2.2, 1.3]
explode=(0.1,0, 0, 0, 0, 0, 0, 0, 0, 0)
ax.pie(percentages, explode=explode, labels=labels, autopct='%1.0f%%', 
       shadow=False, startangle=0,   
       pctdistance=1.2,labeldistance=1.4)
ax.axis('equal')
#ax.set_title("Distribution of Applications by Category")
ax.legend(frameon=False, bbox_to_anchor=(2,0.8))


#Average rating by category
avg_rating_cat = df.groupby(['Category'])['Rating'].mean().sort_values(ascending=False)
avg_rating_cat
avg_rating_cat.plot(title='Average Rating by Category',
                    kind='bar')
plt.show()
sum_rating_cat = df.groupby(['Category'])['No of people Rated'].sum().sort_values(ascending=False)
sum_rating_cat.plot(title='Total Number of People',
                    kind='bar')
plt.show()

df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Day'] = pd.DatetimeIndex(df['Date']).day
df['Day of the Week'] = pd.DatetimeIndex(df['Date']).strftime("%A")




df_year = df.groupby(['Year', 'Category'])['No of people Rated'].sum()
df_year
