# Loading necessary packages:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import calendar
%matplotlib inline
import csv
from pandas import DataFrame
import warnings
warnings.filterwarnings('ignore')
import datetime
sns.set(style="white", color_codes=True)
dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
# Importing data
df = pd.read_csv('../input/911.csv')
# Getting necessary info from the data:
df.info()
for i in df.columns:
    print(i, "\t", df.loc[:, i].isnull().sum()/len(df))
# Since one of the columns (zip code) has almost 12% of missing values, I am not going to use that for further analysis.
reason = np.unique(df['title'])
reason.size
df['type'] = df["title"].apply(lambda x: x.split(':')[0])
df["type"].value_counts()

# The highest being EMS
# Plotting the reasons(categories) for 911 calls:

# adding 'Reasons' column in main df
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df.head()

# Frequencies of each reason
sns.countplot(x='Reason',data=df,palette='cubehelix')

# Pie chart depicting three main reasons(categories) of call:

labels = 'EMS', 'Traffic', 'Fire'
sizes =  161441,116065,48919

colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0.1, 0.1)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
df2 = df['title'].value_counts().head(5)
df2
# Makeing category and sub-category column for each emergency:
df['Category'] = df['title'].apply(lambda x: x.split(':')[0])
df['Sub-Category'] = df['title'].apply(lambda x: ''.join(x.split(':')[1]))
df[df['Category'] == 'EMS']['Sub-Category'].value_counts().head(5)
df[df['Category'] == 'Fire']['Sub-Category'].value_counts().head(5)
df[df['Category'] == 'Traffic']['Sub-Category'].value_counts().head(5)
df[df['Category'] == 'EMS']['twp'].value_counts().head(10)
# Plotting the top ten EMS call towns with frequencies:
plt.figure(figsize = (10,5))
plt.title('Top places for EMS category')
sns.countplot('twp', data = df[(df['Category'] == 'EMS') & (df['twp'].isin(['NORRISTOWN', 'LOWER MERION', 'ABINGTON',
                                                              'POTTSTOWN', 'LOWER PROVIDENCE', 'UPPER MERION', 
                                                              'CHELTENHAM', 'UPPER MORELAND', 'HORSHAM', 
                                                              'PLYMOUTH']))], palette = 'cubehelix')
plt.xticks(rotation = 60)
df[df['Category'] == 'Fire']['twp'].value_counts().head(10)
# Plotting the top ten Fire call towns with frequencies:
plt.figure(figsize = (10,5))
plt.title('Top places for Fire category')
sns.countplot('twp', data = df[(df['Category'] == 'Fire') & (df['twp'].isin(['LOWER MERION', 'ABINGTON',
                                                              'NORRISTOWN','CHELTENHAM','POTTSTOWN','UPPER MERION','WHITEMARSH','UPPER PROVIDENCE','LIMERICK','UPPER MORELAND', 
                                                              ]))], palette = 'cubehelix')
plt.xticks(rotation = 60)
df[df['Category'] == 'Traffic']['twp'].value_counts().head(10)
# Plotting the top ten Traffic call towns with frequencies:
plt.figure(figsize = (12,5))
plt.title('Top places for Traffic category')
sns.countplot('twp', data = df[(df['Category'] == 'Traffic') & (df['twp'].isin(['LOWER MERION','UPPER MERION','ABINGTON',
                                                              'CHELTENHAM','PLYMOUTH','UPPER DUBLIN','UPPER MORELAND','MONTGOMERY','HORSHAM','NORRISTOWN' 
                                                              ]))], palette = 'cubehelix')
plt.xticks(rotation = 60)
# Sub-Category plotting(EMS)
plt.figure(figsize = (15,5))
sns.countplot('Sub-Category', data = df[df['Category'] == 'EMS'], palette = 'cubehelix')
plt.xticks(rotation = 90)
# Sub-Category plotting(Traffic)
plt.figure(figsize = (10,5))
sns.countplot('Sub-Category', data = df[df['Category'] == 'Traffic'], palette = 'cubehelix')
plt.xticks(rotation = 60)
# Sub-Category plotting(Fire)
plt.figure(figsize = (15,5))
sns.countplot('Sub-Category', data = df[df['Category'] == 'Fire'], palette = 'cubehelix')
plt.xticks(rotation = 90)
#Converting the time data set into datetime format
type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['timeStamp'].iloc[0]
#Grabbing the date from this timestamp.
df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)
#Now doing the same for day of weeks:
df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)
# Making day of the week as string:
dmap = {0:'mon', 1:'tue',2:'wed', 3:'thu', 4:'fri', 5:'sat', 6:'sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
# importing date in the data:
df['date'] = df['timeStamp'].apply(lambda t:t)
dayHour = df.groupby(by = ['Day of Week', 'Hour']).count()['Category'].unstack()
dayHour
fig = plt.figure(figsize = (10,7))
sns.heatmap(dayHour, cmap = 'Blues')
