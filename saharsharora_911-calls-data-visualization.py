
# Step 1 : Importing the libraries required for this project. 
import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
#Visualization Libraries: 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# Step 2 : Importing and Reviewing the Dataset: 
df = pd.read_csv('../input/911.csv')
df.head()
df.info()
#Step 3 : Let's get to answering some questions that may be important or relevant! 

# Top 10 zipcodes for 911 calls: 
df['zip'].value_counts().head(10)
# top 10 townships for 911 calls: 
df['twp'].value_counts().head(10)
# Total unique titles:
df['title'].nunique()
# Creating a new Column to separate out the reasons for Title Codes: 
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

#Determining the most common reason: 
df['Reason'].value_counts()


#CREATING VISUALIZATIONS 
# Countplot for Call Reasons: 
sns.countplot(x='Reason',data=df,palette="cubehelix")




#Converting and obtaining DateTime Data: 
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week'] = df['Day of Week'].map(dmap)

sns.countplot(x='Day of Week',data=df,hue='Reason',palette="cubehelix")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# This plot show is that some months are missing, so we will now work on filling that
# Information in. 

byMonth = df.groupby('Month').count()
byMonth.head()
byMonth['twp'].plot()

#Creating a date Column 
df['Date']=df['timeStamp'].apply(lambda t: t.date())

#Plotting a graph of 911 calls using GroupBy and Count on the Date column, 

df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
#Plotting the calls made only due to Traffic related incidents: 
df['Date']=df['timeStamp'].apply(lambda t: t.date())
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
#Creating Heatmaps

#First we will use unstack() in order to restructure the data frame so that the days of the week become the index.

dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')
#Clustermap: 
sns.clustermap(dayHour,cmap='viridis')
