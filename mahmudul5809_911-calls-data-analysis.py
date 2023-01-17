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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# Read in the csv file as a dataframe called df
df = pd.read_csv('/kaggle/input/montcoalert/911.csv')
# Check the info() of the df 
df.info()
# Check the head of df
df.head(5)
# What are the top 5 zipcodes for 911 calls?
df['zip'].value_counts().head(5)
# What are the top 5 townships (twp) for 911 calls?
df['twp'].value_counts().head(5)
# Take a look at the 'title' column, how many unique title codes are there?
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason']
# What is the most common Reason for a 911 call based off of this new column? 
df['Reason'].value_counts()
# Now use seaborn to create a countplot of 911 calls by Reason. 
sns.countplot(x='Reason',data=df,palette='viridis')
#  Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? 
type(df['timeStamp'].iloc[0])
# you should have seen that these timestamps are still strings. Use pd.to_datetime to convert the column from strings to DateTime objects.
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
#  Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Now do the same for Month:
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = df.groupby('Month').count()
byMonth.head()
# Now create a simple plot off of the dataframe indicating the count of calls per month
# Could be any column
byMonth['twp'].plot()
# Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
df['Date']=df['timeStamp'].apply(lambda t: t.date())
# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
plt.figure(figsize=(12,6))
df.groupby('Date').count()['twp'].plot()

plt.tight_layout()

# Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 cal
plt.figure(figsize=(12,6))
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
plt.figure(figsize=(12,6))
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
plt.figure(figsize=(12,6))
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
# Now create a HeatMap using this new DataFrame. 
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')
# Now create a clustermap using this DataFrame. 
sns.clustermap(dayHour,cmap='viridis')
# Now repeat these same plots and operations, for a DataFrame that shows the Month as the column.
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,cmap='viridis')
