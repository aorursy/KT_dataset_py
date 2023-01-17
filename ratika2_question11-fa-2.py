'''

Present appropriate visualizations of your analysis of the “Emergency - 911 Calls” dataset on Kaggle.

This data contains 326k rows and 9 columns. 

The recommended approach here  is  – 

ask questions and answer them using the apt visualizations, tabulations, etc.

Note: Do this analysis only on Traffic related incidents.

'''

# Number of calls over the period of time and by type?
# Questions

'''

I EDA on whole dataset

1.1. reading the dataset using pandas dataframe

1.2. gathering information of type of each varaible in data

1.3. summarizing the data

1.4. Checking for missing values in the dataset.

1.5. Identifying reasons to make ememrgency call and total number of unique calls made

1.6. Visulazing the type of calls made using bar plot



II Analysis of Traffic related incidents

2.1 Subsetting the traffic related calls

2.2 Determimg the reasons for traffic related calls made

2.3. Top 10 towns from where maximum traffic related calls were made

2.4. Visualizaing the time and week days where maximum calls were made using Heatmap

2.5. Visualiziang top 3 traffic related incidents for time and day of week using Heatmap

2.6. Visualizing months for which maximum number of calls made

'''
# 1.1.reading the dataset using pandas dataframe
# reading dataset using pandas library

import pandas as pd

emer =  pd.read_csv("../input/911.csv")

# displaying first 5 rows of dataset

emer.head()
# checking type of each column

emer.info()
# describing/finding summary of  the whole dataset

emer.describe()
# checking for null values 

emer.isnull().any(axis=0)
# Q1. What are the number of missing values in each colum?



for i in emer.columns:

    print(i, "\t", emer.loc[:, i].isnull().sum()/len(emer))
# What are the total number of unique reasons to make 911 calls?

import numpy as np

reason = np.unique(emer['title'])

reason.size
# what are main reasons to call 911

emer['title'].value_counts()
emer['title'].unique()
# splitting type at (:) and totalling reasons category wise to call 911

emer['type'] = emer["title"].apply(lambda x: x.split(':')[0])

emer['type'].value_counts()
#  Plotting the reasons(categories) for 911 calls:



# adding 'Reasons' column in main df

emer['type'] = emer["title"].apply(lambda x: x.split(':')[0])





# Frequencies of each reason

import seaborn as sns

sns.countplot(x='type',data= emer,palette='cubehelix')



# making subset dataframe of 911 call type being traffic

traffic_set = emer[emer['type'] == 'Traffic']

print(traffic_set.head(10))
# reasons for 'Traffic' 911 calls:

emer['Sub-Category'] = emer['title'].apply(lambda x: ''.join(x.split(':')[1]))

emer[emer['type'] == 'Traffic']['Sub-Category'].value_counts().head(10)
t = emer[emer['type'] == 'Traffic']['Sub-Category'].value_counts().plot(kind='barh')

# Top 10 towns where maximum 'Traffic' calls were made from:

emer[emer['type'] == 'Traffic']['twp'].value_counts().head(10).plot(kind ='barh')
## timestamp change, which month me highest occurance of traffic accidents and town simulaneously

# Using timestamp to deduce time, hour, day of the week and month where calls were made

#Converting the time data set into datetime format

type(emer['timeStamp'].iloc[0])

emer['timeStamp'] = pd.to_datetime(emer['timeStamp'])

emer['timeStamp'].iloc[0]



#Grabbing the date from this timestamp.

emer['Hour'] = emer['timeStamp'].apply(lambda time:time.hour)





#Now doing the same for day of weeks:

emer['Day of Week'] = emer['timeStamp'].apply(lambda time:time.dayofweek)





# Making day of the week as string:

dmap = {0:'mon', 1:'tue',2:'wed', 3:'thu', 4:'fri', 5:'sat', 6:'sun'}

emer['Day of Week'] = emer['Day of Week'].map(dmap)





# importing date into emergency dataset :

emer['date'] = emer['timeStamp'].apply(lambda t:t)





traffic_set = emer[emer['type'] == 'Traffic']



dayHour = traffic_set.groupby(by = ['Day of Week', 'Hour']).count()['title'].unstack()

dayHour

fig = plt.figure(figsize = (10,7))

sns.heatmap(dayHour, cmap = 'Blues')
import seaborn as sns

from matplotlib.pyplot import figure, show

traffic_set['month']= traffic_set['timeStamp'].dt.month



width=18

height=8

figure(figsize=(width,height))

sns.countplot(traffic_set['month'],palette="muted",hue = traffic_set['title'])
import numpy as np

 

import seaborn as sb

 

import matplotlib.pyplot as plt







dayHour = traffic_set.groupby(by = ['month', 'twp']).count()['title'].unstack()

dayHour

fig = plt.figure(figsize = (20,8))

sns.heatmap(dayHour, cmap = 'YlGnBu')