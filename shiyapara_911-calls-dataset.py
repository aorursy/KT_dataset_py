# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style ('whitegrid')
df= pd.read_csv('/kaggle/input/montcoalert/911.csv')

df.dataframe = '911.csv'

df.head()
# lat : String variable, Latitude

# lng: String variable, Longitude

# desc: String variable, Description of the Emergency Call

# zip: String variable, Zipcode

# title: String variable, Title

# timeStamp: String variable, YYYY-MM-DD HH:MM:SS

# twp: String variable, Township

# addr: String variable, Address

# e: String variable, Dummy variable
df.info()
df.isnull
df['zip'].value_counts().head(10)
df['twp'].value_counts().head(10)
df['title'].nunique()
# New column "Reason" contains string values (EMS|Fire|Traffic)

df['Reason'] =df['title'].apply(lambda title: title.split (':')[0])

df.head(3)
df.Reason.value_counts()
sns.countplot(x ='Reason', data = df, palette = 'viridis')
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
time= df['timeStamp'].iloc[0]

time.hour
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

df['Year'] = df['timeStamp'].apply(lambda time: time.year)
dmap={0:'Mon', 1:'Tue', 2:'Wed',3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x ='Day of Week', data=df, hue ='Reason', palette = 'deep')

plt.legend(bbox_to_anchor = (1.05, 1), loc =2, borderaxespad =0.)
sns.countplot(x ='Month', data=df, hue ='Reason', palette = 'deep')

plt.legend(bbox_to_anchor = (1.05, 1), loc =2, borderaxespad =0.)
sns.countplot(x ='Year', data=df, hue ='Reason', palette = 'deep')

plt.legend(bbox_to_anchor = (1.05, 1), loc =2, borderaxespad =0.)
byyear= df.groupby('Year').count()

byyear.head(12)
bymonth=df.groupby('Month').count()

bymonth.head(12)
bymonth['twp'].plot(figsize =(10,8))
sns.lmplot(x='Month',y='twp',data=bymonth.reset_index())
df['Date']= df.timeStamp.dt.date

df.Date
byDate= df.groupby('Date').count()

byDate.head()
byDate.twp.plot(figsize=(15,8))
plt.figure(figsize=(15, 8))



df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
plt.figure(figsize=(15, 8))



df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
plt.figure(figsize=(15, 8))



df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()