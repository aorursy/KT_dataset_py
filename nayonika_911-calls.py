import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/911.csv')

df.head()
df.info()
df.describe()
#Top 5 zipcodes 

df['zip'].value_counts().head(5)
#Top 5 Townships

df['twp'].value_counts().head(5)
# Number of unique titles for emergencies

df['title'].nunique()
#Creating a new Reason column based on title to separate the Reason/Department part from title

df['Reason']=df['title'].apply(lambda x: x.split(':')[0])

#Number of rows for each Reason

df['Reason'].value_counts()
#Using a countplot to show the number of Reasons

sns.countplot(x='Reason', data=df)

#Converting the timestamp column into datetime format from str format 

df['timeStamp']=pd.to_datetime(df['timeStamp'])



#creating new columns-Hour, Month and Day of Week to separate the hour, month and day value from timestamp

df['Hour']=df['timeStamp'].apply(lambda time: time.hour)

df['Month']=df['timeStamp'].apply(lambda time: time.month)

df['Day of Week']=df['timeStamp'].apply(lambda time: time.dayofweek)



#The Day of Week column appears as an integer ranging from 0-6, using dmap we map string values to the 

#integers such as follows

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week']=df['Day of Week'].map(dmap)



#Create a countplot to see number of 911 calls on each day of the week for each of the Reasons

sns.countplot(x='Day of Week', data=df, hue='Reason')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Let us see a similar kind of plot for monthly distribution of 911 calls

sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#Heatmap plot for day of week and month

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')