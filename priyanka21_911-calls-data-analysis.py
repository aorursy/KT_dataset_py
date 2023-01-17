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
import pandas as pd
pwd
import os
os.chdir('/kaggle/input/montcoalert')
pwd
df = pd.read_csv('911.csv')
df.info()
df.head()
df['zip'].value_counts().head(5) #top 5 zip codes on 911 calls
df['twp'].value_counts().head(5) #top 5 townships of 911 calls
df['title'].nunique() #how many unique title codes are there
len(df['title'].unique()) #other way of calculating the same thing
df['reason'] = df['title'].apply(lambda title: title.split(':')[0])  #creating a new column that includes the reason/department of the 911 call
df['reason'].value_counts()  #the most common reasons for 911 calls
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.countplot(x='reason',data=df)
plt.title('count of calls by reason')
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'].iloc[0])
df['hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['month'] = df['timeStamp'].apply(lambda x: x.month)
df['day of week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
df.info()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
df['day of week'] = df['day of week'].map(dmap)
type(df['day of week'].iloc[0])
sns.countplot(x='day of week',data=df,hue='reason')
plt.title('count of calls by day of week with hue based off reason')
sns.countplot(x='month',data=df,hue='reason')
plt.title('count of calls by month with hue based off reason')
bymonth = df.groupby('month').count()
bymonth
sns.lmplot(x='month',y='twp',data=bymonth.reset_index())
plt.title('regression line of number of calls per month')
sns.countplot(x='month',data=df)
plt.title('count of calls per month')
df['date'] = df['timeStamp'].apply(lambda x: x.date())
df['date'].head()
bydate = df.groupby('date').count()
bydate['twp'].plot()
plt.tight_layout()
df[df['reason']=='EMS'].groupby('month').count().plot()
plt.title('EMS calls group by month')
plt.tight_layout()
df[df['reason']=='Fire'].groupby('month').count().plot()
plt.title('Fire calls group by month')
plt.tight_layout()
df[df['reason']=='Traffic'].groupby('month').count().plot()
plt.title('Traffic calls group by month')
plt.tight_layout()
