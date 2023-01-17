# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
# top 5 zipcodes for 911 calls
df['zip'].value_counts().head(5)
# he top 5 townships (twp) for 911 calls
df['twp'].value_counts().head(5)
# number of unique title code in the data set
df['title'].nunique()
'''
creating a new column "reason" to specify the reason to call 911
and finding out the most commom reason for the same
'''

df['Reason'] = df['title'].apply(lambda title: title.split(":")[0]) #creating a new column 'reason'
df['Reason'].value_counts()  #getting the most common reason for calling 911
# use seaborn to create a countplot of 911 calls by Reason.

sns.countplot(x='Reason',data=df,palette='viridis')
#converting the data type of timeStamp into pd.to_datetime format

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
#grab specific attributes from a Datetime object by calling them eg:
time = df['timeStamp'].iloc[0]

time.hour
# creating 3 new columns called Hour, Month, and Day of Week.

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
# the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
#creating a countplot of the Day of Week column with the hue based off of the Reason column
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # To relocate the legend
#creating a countplot of the Month with the hue based off of the Reason column

sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # To relocate the legend
'''
if we see in the above graph there are some months missing. 
we create a groupby object called  "byMonth" where you group the DataFrame by the month column and use the count() method for aggregation. 
Use the head() method on this returned DataFrame

'''
byMonth = df.groupby('Month').count()
byMonth.head()

# create a simple plot off of the dataframe indicating the count of calls per month.
byMonth['twp'].plot()
'''
seaborn's lmplot() to create a linear fit on the number of calls per month. 
Keep in mind you may need to reset the index to a column.
'''
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
'''
Create a new column called 'Date' that contains 
the date from the timeStamp column. You'll need to use apply along with the .date() method.
'''
df['Date']=df['timeStamp'].apply(lambda t: t.date())
# roupby this Date column with the count() aggregate and create a plot of counts of 911 calls.
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
# 3 separate plots with each plot representing a Reason for the 911 call

df[df['Reason'] == "Traffic"].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()

df[df['Reason'] == "Fire"].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()

df[df['Reason'] == "EMS"].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
# creating heatmap with the data frame
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')
# creating clustermap from the data frame

sns.clustermap(dayHour,cmap='viridis')
'''
DataFrame that shows the Month as the column.
'''
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()

plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,cmap='viridis')
