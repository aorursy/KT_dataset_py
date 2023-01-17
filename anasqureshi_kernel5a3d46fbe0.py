import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

cal = pd.read_csv('../input/911.csv')

cal.info()
cal.head(3)
#top 5 zip codes for 911

cal['zip'].value_counts().head(5)
#top 5 township(twp) for 911 calls

cal['twp'].value_counts().head(5)

#number of unique column

cal['title'].nunique()
cal['title']
def ref_string(code):

    if "Fire" in code:

        return "Fire"

    elif "EMS" in code:

            return "EMS"

    elif "Traffic" in code:

                return "Traffic"

    else:

                return False

            



cal['Reason'] = cal['title'].apply(lambda x: ref_string(x))

cal['Reason']

#most common reason for 911 calls

cal['Reason'].max()
#use seaborn to create a countplot for 911 calls by reason 

import seaborn as sns

sns.countplot(x = 'Reason',data =cal)
cal.head(2)
cal.dtypes
#convert timeStamp column strings to date and time objects

cal['timeStamp'] = pd.to_datetime(cal['timeStamp'],errors = 'coerce')

cal.info()
time = cal['timeStamp'].iloc[0]

time.dayofweek
#create 3 new columns called hour, month and day of week 

cal['hour'] = cal['timeStamp'].apply(lambda time: time.hour)

cal['month'] = cal['timeStamp'].apply(lambda time: time.month)

cal['Day of Week'] = cal['timeStamp'].apply(lambda time: time.dayofweek)



cal['month']

cal['hour']

cal['Day of Week']



cal.head(2)
cal.drop(['day'],axis=1)
cal['Day of Week'] = cal['timeStamp'].apply(lambda time: time.dayofweek)
cal.dtypes
dmap = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}

cal['Day of Week'] = cal['Day of Week'].map(dmap)
cal['Day of Week']
#countplot of the day of week column with the hue based off the reason column

import seaborn as sns

sns.countplot(x='Day of Week',data = cal, hue = 'Reason',palette = 'viridis')
cal['month']
import seaborn as sns

sns.countplot(x='month',data = cal, hue = 'Reason',palette = 'viridis')
byMonth = cal.groupby('month').count()

byMonth.head()
byMonth['lat'].plot()
#create a column that contains date only 

t = cal['timeStamp'].iloc[0]

t.date()
cal['date'] = cal['timeStamp'].apply(lambda t: t.date())

cal.head()

x = cal.groupby('date').count().head()
x['lat'].plot()
hr = cal.groupby(by=['Day of Week','hour']).count()['Reason'].unstack()

print(hr)
sns.heatmap(hr,cmap = 'viridis')
