import numpy as np
import pandas as pd
df = pd.read_csv('../input/montcoalert/911.csv')
df.head()

df.info()
df['zip'].value_counts().head() #counting the 5 Top Zip codes
df['twp'].value_counts().head() #Count for call made Township wise
df['title'].nunique() #Unique no of titles
df
df['Reason']= df['title'].apply(lambda x: x.split(":")[0])
df.head()  #Isolating the Reason Tab for better exploratory analysis
df['Reason'].value_counts() #Counting the no of occurences
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Reason',data=df)  #Countplot of the reason column

type(df['timeStamp'].iloc[0])
df['timeStamp']=pd.to_datetime(df['timeStamp']) #Converting the string datatype into datetime datatype
df['timeStamp'] 
type(df['timeStamp'].iloc[0])
time = df['timeStamp'].iloc[0]
time
#time.hour
#time.month
#time.day
#time.dayofweek  #Attributes of the new datatype
df['Hour']=df['timeStamp'].apply(lambda x: x.hour)
df['Hour'].value_counts()  #Creating a new column 'Hour'
df['Month']=df['timeStamp'].apply(lambda x: x.month)
df['Month'].value_counts() #Creating a new column Month
df['DayOfWeek']=df['timeStamp'].apply(lambda time: time.dayofweek) #Creating a new column DayOfWeek
df['DayOfWeek'].value_counts()
type(df['DayOfWeek'].iloc[0])
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['DayOfWeek']=df['DayOfWeek'].map(dmap) #Mapping the integer values into day names
df

df.head()
sns.countplot(x='DayOfWeek',data=df,hue='Reason')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0) #Countplot referencing the reason column for approx count of the day of week column
sns.countplot(x='Month',data=df,hue='Reason')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0) #Countplot referencing the reason column for approx count of the Month column
df1=df.groupby(by='Month').count() #Grouping the Data Frame by Month
df1.head()
df1['twp'].plot(grid=True) #Plot shoing the number of calls made from each township during a particular month

%matplotlib inline
sns.set_style('whitegrid')
sns.lmplot(x='Month',y='twp',data=df1.reset_index()) #Creating a best fit plot
plt.tight_layout()
df['timeStamp'].loc[0].date()
df['Date']=df['timeStamp'].apply(lambda t: t.date()) #Creating a new Date column
df2=df.groupby('Date')
df2.count()['twp'].plot() #Creating a plot to display the number of calls made per township on the mentioned dates
plt.tight_layout()

df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot() #Creating a plot to display the number of traffic related calls
plt.title('Traffic')
plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot() #Creating a plot to display the number of fire related calls
plt.title('Fire')
plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot() #Creating a plot to display the number of EMS related calls
plt.title('EMS')
plt.tight_layout()
df.head()
dayHour = df.groupby(by=['DayOfWeek','Hour']).count()['Reason'].unstack() #Grouping by DayofWeek as Index and Hour as Column
dayHour.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis') #heatmap for the new created groupby Data Frame
sns.clustermap(dayHour,cmap='viridis') #Clustermap
dayMonth = df.groupby(by=['DayOfWeek','Month']).count()['Reason'].unstack() #Creating a new dataframe by grouping dayofweek and month
dayMonth.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis') #Heatmap
sns.clustermap(dayMonth,cmap='viridis') #Clustermap