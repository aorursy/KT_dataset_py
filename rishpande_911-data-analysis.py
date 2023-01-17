import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
df = pd.read_csv("../input/911.csv")
df.head(5)
df.shape
df.describe()
# Top 5 zipcodes for 911 calls:

df['zip'].value_counts().head(5)
#top 5 townships for 911 calls? 
df['twp'].value_counts().head(5)

df['title'].nunique() #unique title codes
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason'].value_counts() #most common Reason for a 911
plt.figure(figsize=(15,8))

sns.countplot(x='Reason',data=df,palette='viridis') # countplot of 911 calls by Reason. 

type(df['timeStamp'].iloc[0]) # data type of the objects in the timeStamp column
# Use pd.to_datetime to convert the column from strings to DateTime objects. **

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#grab specific attributes from a Datetime object by calling them

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
plt.figure(figsize=(15,8))

sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis') #ountplot of the Day of Week column with the hue based off of the Reason column.

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure(figsize=(15,8))

sns.countplot(x='Month',data=df,hue='Reason',palette='viridis') #same for month

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
#gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation
byMonth = df.groupby('Month').count()
byMonth.head()
# Could be any column
plt.figure(figsize=(18,10))
byMonth['twp'].plot()
plt.show()
sns.lmplot(x='Month',y='twp',height = 12, data=byMonth.reset_index()) # number of calls per month

#new column called 'Date' that contains the date from the timeStamp column
df['Date']=df['timeStamp'].apply(lambda t: t.date()) 
plt.figure(figsize=(18,10))
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()

plt.figure(figsize=(18,10))
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
plt.figure(figsize=(18,10))


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
plt.figure(figsize=(18,10))

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')

plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
#HeatMap using this new DataFrame
plt.figure(figsize=(18,10))

sns.heatmap(dayHour,cmap='viridis')
plt.figure(figsize=(18,10))

sns.clustermap(dayHour,figsize=(16, 17), cmap='viridis') #clustermap
# repeat these same plots and operations, for a DataFrame that shows the Month as the column
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(18,10))
sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,figsize=(16, 17),cmap='viridis')
