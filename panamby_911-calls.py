# First of all, I will import the two libraries that will help me analyze the 911 data. 
# One of them is the poweful Pandas.🐼

import numpy as np
import pandas as pd
# Secondly, I will import the libraries for data visualization.
# For a greater practicality, I will tap a code which will enable to watch the graphs after coding them.

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
# Now I will read the file that I will analyze and visualize.
df = pd.read_csv('../input/911.csv')
# Let's extract some informations from this file.

df.info()   # We can see that there are more than 300,000 entries and 9 columns. 
            # There are numbers (floats and integers) and strings.
# Let's check the head!

df.head()   # The data contains the following fields:

            # lat : String variable, Latitude
            # lng: String variable, Longitude
            # desc: String variable, Description of the Emergency Call
            # zip: String variable, Zipcode
            # title: String variable, Title
            # timeStamp: String variable, YYYY-MM-DD HH:MM:SS
            # twp: String variable, Township
            # addr: String variable, Address
            # e: String variable, Dummy variable (always 1)
# Now I will answer some basic questions!
# What're the top 5 zipcodes for 911 calls?

df.zip.value_counts().head(5)
# What're the top 5 townships (twp) for 911 calls?

df.twp.value_counts().head(5)
# How many unique title codes are there?

df.title.nunique()
# To better understand the informations whitin the data and posteriorly to use for a better prediction, ...
# ... is important create a new column from 'title' column with only the Reason.

df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
# Lets check the result!

df.head(3)
# What's the most common Reason for a 911 call based in this new column?

df.Reason.value_counts()
# Let's do the first graph visualization.
# "911 calls X Reason""

sns.countplot(x='Reason', data=df)
# What's the data type of the objects in the 'timeStamp' column?

type(df['timeStamp'].iloc[0])
# The timestamps are still strings. 
# I have to convert the column to DateTime objects with the purpose to better use in data analize.

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
# One more time, to good understand the informations and get a better prediction, ...
# ... is important create new columns, from 'timeStamp' column, called 'Hour', 'Month' and 'Day of Week'.

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

# Let's see the result.

df.head(3) 
# The 'Day of Week' column has only number instead of strings, such as 'Mon', 'Sun', 'Thu' and so on.
# I will solve this problem.

dmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)

# Let's check if the problem was solved.

df.head(3) # Now we can see that the values in the 'Day of Week'changed from number to days of week.
# Now I can use these new columns to do a new graph using 'Day of Week' column with the hue based in the Reason column.

sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Now I will do the same thing for 'Month'

sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 

# We can notice that is missing some months: 9,10, and 11 are not there.
# let's see if we can maybe fill in this information by plotting the information in another way.
# I will create a gropuby object called 'byMonth', where I group the DataFrame by the month column and use the count() method for aggregation.

byMonth = df.groupby('Month').count()

# Let's chck the result!

byMonth.head()
# Now I will create a simple plot of the dataframe indicating the count of calls per month.
# Could be any column

byMonth['twp'].plot() #That's it. I got it!
# I will see if I can use seaborn's lmplot() to create a linear fit on the number of calls per month. 
# I can't forget that I need to reset the index to a column.


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
# I will create a new column called 'Date' that contains the date from the timeStamp column.
# As I sad before, these news columns helps better unsderstand the informations.

df['Date']=df['timeStamp'].apply(lambda t: t.date())

# Let's check!

df.head(3)
# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.

plt.figure(figsize=(14,6)) # To better distinguish the legends.
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
# Now I will recreate this plot, but creating 3 separate plots with each plot representing a Reason for the 911 call.

plt.figure(figsize=(14,6))
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
plt.figure(figsize=(14,6))
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
plt.figure(figsize=(14,6))
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()

# we can conclude that the 'Fire' has the lowest rate of call and 'EMS' has the uppest.  
# Now let's move on to creating heatmaps with seaborn and our data. 
# We'll first need to restructure the dataframe, so that the columns become the Hours and the Index becomes the Day of the Week.

dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

# Let'see...
dayHour.head()
# Now I will create a HeatMap using this new DataFrame.

plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')

# We can notice that the higher volume of call happens Wednesday and Friday between 15h and 17h.
# Now I will create a clustermap using this DataFrame.


sns.clustermap(dayHour,cmap='viridis')

# We can notice better is this clustermap graph that the higher volume of call happens only on Friday between 15h and 17h.
# Now I will repeat these same plots and operations for Month as the column.
# I'm trying to get more and better informations and visualizations from data.

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

# Let's check!

dayMonth.head()
# Heatmap

plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')

# We can notice that 'Friday' appears again, but, joining the informations from the last heatmap, the higher calls happens in March, on friday and between 15h/17h.
# Clustermap

sns.clustermap(dayMonth,cmap='viridis')

# This map reinforce the last conclusion.