
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#Import our data analysis/visualization libraries along with any other useful libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


#read in data
df_calls = pd.read_csv('../input/911.csv')
df_calls.info()
df_calls.head()
df_calls.isnull().sum()
#visualize null values
sns.heatmap(df_calls.isnull(),cmap = 'plasma')
#Visualization of number of calls relative to zip code

df_calls['zip'].value_counts().head(10).plot.bar(color = 'green')
plt.xlabel('Zip Codes',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with Most Calls')
#Visualization of number of calls relative to township

df_calls['twp'].value_counts().head(10).plot.bar(color = 'teal')
plt.xlabel('Townships', labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Townships with Most Calls')
df_calls['title'].head(3)
#New columns that extract call info from title column to use for further analysis
df_calls['Reason'] = df_calls['title'].apply(lambda x: x.split(':')[0])
df_calls['Emergency Description'] = df_calls['title'].apply(lambda x: x.split(':')[1])
# Function to remove hyphen at end of values for 'Emergency Description'

def hyph_del(x):
    if x[-1] == '-':
        return x[:-2]
    else:
        return x

df_calls['Emergency Description'] = df_calls['Emergency Description'].apply(hyph_del)
#gives count of reason type
df_calls['Reason'].value_counts()
#Orders top 30 description calls
df_calls['Emergency Description'].value_counts().head(30)
sns.countplot('Reason', data=df_calls, palette='pastel')
df_calls['Reason'].value_counts()
df_calls['Emergency Description'].value_counts().head(20).plot.bar(color = 'navy')
plt.xlabel('Emergency Description',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Top 20 Emergency Description Calls')
#Converts string in timeStamp to datetime object
df_calls['timeStamp'] = pd.to_datetime(df_calls['timeStamp'])
df_calls['Hour'] = df_calls['timeStamp'].apply(lambda time: time.hour)
df_calls['Month'] = df_calls['timeStamp'].apply(lambda time: time.month)
df_calls['Day of Week'] = df_calls['timeStamp'].apply(lambda time: time.dayofweek)
#Change Day of Week column from integer to string by mapping values to string
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df_calls['Day of Week'] = df_calls['Day of Week'].map(dmap)
mmap = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',
       8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

df_calls['Month'] = df_calls['Month'].map(mmap)
order = ['Sun','Mon', 'Tue', 'Wed','Thu','Fri','Sat']

plt.figure(figsize=(10,5))
sns.countplot('Day of Week', data = df_calls, hue = 'Reason', palette='pastel',order = order )
plt.legend(loc= 'upper right', bbox_to_anchor=(1.15,.8))
m_order = ['Jan','Feb', 'Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(10,5))
sns.countplot('Month', data = df_calls, hue = 'Reason', palette='Set2',order = m_order)
plt.legend(loc= 'upper right', bbox_to_anchor=(1.15,.8))
# New column which will use entire date as opposed to seperating the information like before
df_calls['Date'] = df_calls['timeStamp'].apply(lambda time:time.date())
plt.figure(figsize=(15,6))
plt.title('Traffic')
plt.ylabel('Number of Calls')
df_calls[df_calls['Reason'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.tight_layout
plt.figure(figsize=(15,6))
plt.title('Fire')
plt.ylabel('Number of Calls')
df_calls[df_calls['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot(color='green')
plt.tight_layout
plt.figure(figsize=(15,6))
plt.title('EMS')
df_calls[df_calls['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot(color='maroon')
plt.tight_layout
DoW =['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri','Sat']

df_heatHour = df_calls.groupby(by = ['Day of Week', 'Hour']).count()['Reason'].unstack()
df_heatHour.index = pd.CategoricalIndex(df_heatHour.index, categories=DoW)
df_heatHour.sort_index(level=0, inplace=True)
df_heatHour.head()
plt.figure(figsize=(10,7))
sns.heatmap(df_heatHour, cmap='viridis')
plt.title('Relationship of calls between Hour and DoW')
# New column for month as an integer
df_calls['Month_Num'] = df_calls['timeStamp'].apply(lambda time: time.month)

df_heatMonth = df_calls.groupby(by = ['Day of Week', 'Month_Num']).count()['Reason'].unstack()
df_heatMonth.index = pd.CategoricalIndex(df_heatMonth.index,categories = DoW)
df_heatMonth.sort_index(level=0, inplace=True)
df_heatMonth.rename(columns = mmap,inplace=True)
df_heatMonth.head()
plt.figure(figsize=(10,5))
sns.heatmap(df_heatMonth, cmap='viridis')
plt.xlabel('Month')
plt.title('Relationship of calls between Month and DoW')