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
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
calls_data = pd.read_csv('../input/911.csv')
calls_data.info()
calls_data.head()
calls_data.tail()
calls_data['zip'].value_counts().head(5)
calls_data['twp'].value_counts().head(5)
calls_data['title'].nunique()
calls_data['Reason'] = calls_data['title'].apply(lambda x : x.split(':')[0])
calls_data['Reason'].value_counts()
sns.countplot(calls_data['Reason'])
type(calls_data['timeStamp'][0])
calls_data['timeStamp'] = pd.to_datetime(calls_data['timeStamp'])
calls_data['Hour']=calls_data['timeStamp'].apply(lambda x :x.hour)
calls_data['Month']=calls_data['timeStamp'].apply(lambda x :x.month)
calls_data['Day of Week']=calls_data['timeStamp'].apply(lambda x :x.dayofweek)
monthmap = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
calls_data['Month']=calls_data['Month'].map(monthmap)
daymap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
calls_data['Day of Week']=calls_data['Day of Week'].map(daymap)
calls_data['Month'].unique()
calls_data['Day of Week'].unique()
sns.countplot(calls_data['Day of Week'],hue=calls_data['Reason'])
sns.countplot(calls_data['Month'],hue=calls_data['Reason'])
byMonth = calls_data.groupby('Month').count()
byMonth.head()
byMonth['twp'].plot()
calls_data['Date']=calls_data['timeStamp'].apply(lambda t: t.date())
calls_data.groupby('Date').count()['twp'].plot()
plt.tight_layout()
calls_data[calls_data['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
calls_data[calls_data['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
calls_data[calls_data['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
dayHour = calls_data.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth = calls_data.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='rainbow')
sns.clustermap(dayMonth,cmap='rainbow')