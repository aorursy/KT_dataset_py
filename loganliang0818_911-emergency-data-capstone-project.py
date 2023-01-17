import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/911.csv')
df.info()
df.head(5)
df['zip'].value_counts().head()
df['twp'].value_counts().head()
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df['Reason'].value_counts()
sns.countplot(x = 'Reason', data = df, palette = 'viridis')
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
df['Day of Week'].head()
dmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x = 'Day of Week', data = df)
plt.legend(bbox_to_anchor = (1,1))
plt.title('911 Calls per Day of Week')
sns.countplot(x = 'Day of Week', data = df, hue = 'Reason')
plt.legend(bbox_to_anchor = (1,1))
sns.countplot(x='Month',data = df)
plt.legend(bbox_to_anchor = (1,1))
sns.countplot(x='Month',data = df,hue = 'Reason')
plt.legend(bbox_to_anchor = (1,1))
byMonth = df.groupby('Month').count()
byMonth.head()
byMonth['twp'].plot()
sns.lmplot(x = 'Month', y='twp', data = byMonth.reset_index())
df['Date'] = df['timeStamp'].apply(lambda x: x.date())
df.head()
df.groupby('Date').count()['twp'].plot.line(figsize = (15,4))
plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot.line(figsize = (15,4))
plt.title('Traffic')
plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot.line(figsize = (15,4))
plt.title('Fire')
plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot.line(figsize = (15,4))
plt.title('EMS')
plt.tight_layout()
