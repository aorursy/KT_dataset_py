import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Read csv file

#df = pd.read_csv('911.csv')

df = pd.read_csv("../input/911.csv")

df.head(5)
df.info()
# ** What are the top 5 zipcodes for 911 calls? **

df['zip'].value_counts().head(5)
# What are the top 5 townships (twp) for 911 calls?

df['twp'].value_counts().head(5)
# Take a look at the 'title' column, how many unique title codes are there?

df['title'].nunique()
# Take a look at the 'title' column, how many unique title codes are there?

len(df['title'].unique())
df['title'].head()
df['Reason'] = df['title'].apply(lambda x: str(x).split(':')[0])

df['Reason'].head()
# What is the most common Reason for a 911 call based off of this new column?

df['Reason'].value_counts()
# Now use seaborn to create a countplot of 911 calls by Reason.

sns.countplot(x="Reason",data=df,palette='viridis')
type(df['timeStamp'].values)
type(df['timeStamp'][0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['timeStamp'].head()
type(df['timeStamp'].head()[0])
#df['timeStamp'][4].month
df['hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['hour'].head()
df['month'] = df['timeStamp'].apply(lambda x: x.month)

df['month'].head()
dmap = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

df['day'] = df['timeStamp'].apply(lambda x: dmap[x.dayofweek])

df['day'].value_counts()
# Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column.

sns.set_style("whitegrid")

sns.countplot(x='day',data=df,hue='Reason',palette='viridis')

plt.tight_layout()

plt.legend()

# To relocate legend

#plt.legend(bbox_to_anchor_(1.05,1), loc=2, borderaxespad=0.)
# Now use seaborn to create a countplot of the month column with the hue based off of the Reason column.

sns.set_style("whitegrid")

sns.countplot(x='month',data=df,hue='Reason')

plt.tight_layout()

plt.legend()
# Noticed it was missing some Months [9,10,11] see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...

# Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame.

groupBy_month = df.groupby(by='month').count()

groupBy_month
groupBy_month['e']
# Simple plot off of the dataframe indicating the count of calls per month.

groupBy_month['e'].plot(kind='line')
# Similar method

groupBy_month['e'].plot()
# seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column.

#groupBy_month.plot(kind='lmplot')
# Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method.

df['timeStamp'][0].time()
df['date'] = df['timeStamp'].apply(lambda x: x.time())

df['date'].head()
#** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

df.groupby(by='date').count().head()
# In order to plot lmplot by month using groupby data set, we will need to reset index()

groupBy_month.reset_index().head()

#sets month as column now instead of index, we cannot use index to plot linear model plot
# Seaborn lmplots on groupby datasets

sns.lmplot(x='month',y='twp',data=groupBy_month.reset_index())
df.head()

df['date_new'] = df['timeStamp'].apply(lambda t: t.date())
df.head()
# Aggregate date column

groupBy_date = df.groupby(by='date_new').count()

groupBy_date.head()
# plot calls made per day

groupBy_date['e'].plot()
# plot calls made per reason - 3 plots on time series

#groupBy_date.head(1)

# instead of whole data, I will do conditional selection then groupby to plot

# EMS

df[df['Reason'] == "EMS"].groupby(by='date_new').count()['e'].plot()
# Traffic

df[df['Reason'] == "Traffic"].groupby(by='date_new').count()['e'].plot()
# Fire

df[df['Reason'] == "Fire"].groupby(by='date_new').count()['e'].plot()
# USE GroupBy and Unstack Method to create table by desired row vs columns - essentially for heatmaps

df.groupby(by=['day','hour']).count().head()

# Multi-Level Index Created

df.groupby(by=['day','hour']).count().head()['Reason']

# Calling unstack on multilevel group by breaks down it into row vs column, however we can have only one value

df.groupby(by=['day','hour']).count()['Reason'].unstack()
# Heat Maps

plt.figure(figsize=(12,6))

sns.heatmap(df.groupby(by=['day','hour']).count()['Reason'].unstack(),cmap='viridis')
# Cluster Map

plt.figure(figsize=(12,6))

sns.clustermap(df.groupby(by=['day','hour']).count()['Reason'].unstack(),cmap='magma')
# Plt day of week // Month 911 call counts

plt.figure(figsize=(12,6),dpi=100)

day_of_week_month = df.groupby(by=['day','month']).count()['e'].unstack()

sns.heatmap(data=day_of_week_month,cmap='magma',linewidths=0.0001, linecolor='white')
# Plt day of week // Month 911 call counts

plt.figure(figsize=(12,6),dpi=100)

day_of_week_month = df.groupby(by=['day','month']).count().unstack()

sns.clustermap(data=day_of_week_month,cmap='magma',linewidths=0.0001, linecolor='white')