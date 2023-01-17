# Importing numpy and pandas libraries



import numpy as np

import pandas as pd
#Importing Visualization libraries



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Read in the csv file from Kaggle and create a dataframe called df



df=pd.read_csv('../input/montcoalert/911.csv')
#Check the info() of the df



df.info()
#Read in the csv file as a dataframe called df



df.head()
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
#What is the most common Reason for a 911 call based off of this new column?



df['Reason'].value_counts()
#Now using seaborn to create a countplot of 911 calls by Reason.



sns.countplot(x='Reason',data=df,palette='coolwarm')
#Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?



type(df['timeStamp'].iloc[0])
#Use [pd.to_datetime] to convert the column from strings to DateTime objects



df['timeStamp'] = pd.to_datetime(df['timeStamp'])
# Since the timestamp column are actually DateTime objects, we will use .apply() to create 3 new columns called Hour, Month, and Day of Week. 



df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
#Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week:



dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
#Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.



sns.countplot(x='Day of Week',data=df,hue='Reason',palette='coolwarm')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#Now use seaborn to create a countplot of the Month column.



sns.countplot(x='Month',data=df,hue='Reason',palette='coolwarm')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method



df['Date']=df['timeStamp'].apply(lambda t: t.date())
#Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls and recreate this plot representing a Reason for the 911 call



df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call



df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call



df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
# Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. 

#There are lots of ways to do this, but I would recommend trying to combine groupby with an unstack method. 



dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
#Now create a HeatMap using this new DataFrame.



plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='coolwarm')
#Now create a clustermap using this DataFrame



sns.clustermap(dayHour,cmap='coolwarm')
#Now repeat these same plots and operations, for a DataFrame that shows the Month as the column



dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='coolwarm')
sns.clustermap(dayMonth,cmap='coolwarm')