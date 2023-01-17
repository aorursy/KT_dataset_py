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
sns.set_style('darkgrid')
%matplotlib inline
import os
calls = pd.read_csv('../input/911.csv')
calls
calls.isnull().sum()
# Our data has some missing values that may affect our analysis.
# adding column 'Reason'
calls['Reason'] = calls['title'].apply(lambda x: x.split(':')[0])
calls.head()
#reason counts and its plots
df=calls['Reason'].value_counts()
print(df)
sns.countplot(x='Reason',data=calls,palette='viridis')
calls['zip'].value_counts().head(10).plot.bar(color = 'black')
plt.xlabel('Zip Codes',labelpad = 22)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with Most 911 Calls')
# zipcode 19401 has most number of complaints!!
#Visualization of number of calls relative to township

calls['twp'].value_counts().head(10).plot.bar(color = 'orange')
plt.xlabel('Townships', labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Townships with Most 911 Calls')
# Let's split the column 'title', make a column 'emergency' and see how many calls were there for what emergency.
calls['Emergency'] = calls['title'].apply(lambda x: x.split(':')[1])
calls['Emergency'].value_counts().head(30)
#Hence most 911 calls were for 'Vehicle Accident'.
#Visualization of top 10 911 Calls
calls['Emergency'].value_counts().head(10).plot.bar(color = 'red')
plt.xlabel('Emergency',labelpad = 22)
plt.ylabel('Number of 911 Calls')
plt.title('Top 10 Emergency Description Calls')
calls['timeStamp'] = pd.to_datetime(calls['timeStamp'])               # coverting from strings to datetime object
calls['Month'] = calls['timeStamp'].apply(lambda time: time.month)    # creating column"Month'
sns.countplot(x='Month',data=calls,hue='Reason',palette='nipy_spectral')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = calls.groupby('Month').count()
byMonth.head(12)
byMonth['twp'].plot()
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
calls['hour'] = calls['timeStamp'].map(lambda x: x.hour)

groupByMonthDay = calls[(calls['hour'] >= 8) & (calls['hour'] <= 18)].groupby('Month',as_index = False).sum()

yy = groupByMonthDay['e'].values
labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']
xx = groupByMonthDay['Month'].values
width = 1/1.5
plt.bar(xx, yy, width, color="black",align='center')
plt.title('911 Calls each month 8 am to 6 pm')
plt.xticks(xx, labels)
plt.show()
groupByMonthNight = calls[(calls['hour'] > 18) | (calls['hour'] < 8)].groupby('Month',as_index = False).sum()

groupByMonthNight.head()

y = groupByMonthNight['e'].values
labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']
x = groupByMonthNight['Month'].values
width = 1/1.5
plt.bar(x, y, width, color="blACK",align='center')
plt.title('911 Calls each month 6 pm to 8 am')
plt.xticks(x, labels)
plt.show()