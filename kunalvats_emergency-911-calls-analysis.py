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
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()
df = pd.read_csv("../input/911.csv")
df.info()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head()
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df['Reason'].head()
df['Reason'].value_counts()
plt.style.use('ggplot')
df['Reason'].value_counts().iplot(kind='bar')
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
dmap= {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week']= df['Day of Week'].map(dmap)
plt.figure(figsize=(12,8))
sns.countplot(x='Day of Week',data=df,hue='Reason')
plt.legend(loc=[0,1])
plt.title('Day wise count plot for different reasons')
plt.figure(figsize=(12,8))
sns.countplot(x='Month',data=df,hue='Reason')
plt.legend(loc=[0,1])
byMonth = df.groupby('Month').count()
byMonth['Mon'] = (['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
byMon=byMonth.set_index('Mon')
byMon
byMon['twp'].iplot(title='Variation according to Month', xTitle='Month',yTitle='Count',colors='red',width=2)
byMonth.reset_index(inplace=True)
sns.lmplot(y='twp',x='Month',data=byMonth)
df['Date'] = df['timeStamp'].apply(lambda x : x.date())
plt.figure(figsize=(12,8))
df.groupby('Date').count()['twp'].iplot()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].iplot(title='EMS')
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].iplot(title='Traffic')
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].iplot(title='Fire')
new=df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
new
new.iplot(kind='heatmap',xTitle='Days of Week',yTitle="Hour")
sns.clustermap(new,cmap='viridis')
