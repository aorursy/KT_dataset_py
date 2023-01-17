# importing all the packages
# import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#read data file
df_911 = pd.read_csv("../input/911.csv")
df_911.head(10)
# Removing the dummy column "e
df_911.drop(['e'], axis =1, inplace = True)
# Check Column Names
print(df_911.columns)
# Check datatypes of the columns
df_911.dtypes
# Check summary
df_911.describe()
#1. Checking fot NULLS
df_911.isnull().sum()
df_911 = df_911.dropna()
#1. Checking fot NULLS
df_911.isnull().sum()
#2. Checking fot Whitespaces
np.where(df_911.applymap(lambda x: x == ' '))
#VDA
df_911.head(5)
#Most number of calls
df_911['zip'].value_counts().head(10).plot.bar(color = 'blue')
plt.xlabel('Zip Codes',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with Most Calls')
#Least number of calls

df_911['zip'].value_counts().tail(10).plot.bar(color = 'blue')
plt.xlabel('Zip Codes',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with least Calls')
#New columns that extract call info such that call category and description from title column
df_911['callCategory'] = df_911['title'].apply(lambda x: x.split(':')[0])
df_911['callDescription'] = df_911['title'].apply(lambda x: x.split(':')[1])
#To find the no. of calls w.r.t. different call categories
df_911['callCategory'].value_counts()
#1. To plot the number of calls received for the 3 different call categories
sns.countplot('callCategory', data=df_911, palette='pastel')
sns.despine()
#To find the no. of calls w.r.t. different call description
df_911['callDescription'].value_counts().head(30)
#2. To plot the number of calls received for the different call description
df_911['callDescription'].value_counts().head(20).plot.bar(color = 'navy')
plt.xlabel('Call Description',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Top 20 Call Descriptions')
# New column which will use fetch date
df_911['timeStamp'] = pd.to_datetime(df_911['timeStamp'])
df_911['Date'] = df_911['timeStamp'].apply(lambda time:time.date())
#1. No. of calls for EMS category
plt.figure(figsize=(15,6))
plt.title('EMS')
df_911[df_911['callCategory'] == 'EMS'].groupby('Date').count()['lat'].plot(color='maroon')
plt.tight_layout
#2. No. of calls for fire category
plt.figure(figsize=(15,6))
plt.title('Fire')
plt.ylabel('Number of Calls')
df_911[df_911['callCategory'] == 'Fire'].groupby('Date').count()['lat'].plot(color='green')
plt.tight_layout
#3. No. of calls for traffic category
plt.figure(figsize=(15,6))
plt.title('Traffic')
plt.ylabel('Number of Calls')
df_911[df_911['callCategory'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.tight_layout
