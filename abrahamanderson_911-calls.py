# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv("../input/911.csv")

df.info() # here we get overal information about the data
df.head()# here we get the first 5 rows of the data
#here we get the top 5 zipcodes for 911 calls: 

df["zip"].value_counts().head(5)
# here is the top 5 townships (twp) for 911 calls?

df["twp"].value_counts().head(5)
#In the titles column there are "Reasons/Departments" specified before the title code as EMS, Fire, and Traffic. 

#We will use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.

df["Reason"]=df["title"].apply(lambda title: title.split(":")[0])

df.head(10)
#here is the most common Reason for a 911 call based off of this new column:

df["Reason"].value_counts().head(1)
# here we reate a countplot of 911 calls by Reason.

plt.figure(figsize=(20,10))

sns.countplot(x="Reason",data=df,palette="viridis")
# here is the data type of the objects in the timeStamp column:

df["timeStamp"].iloc[0] # it is a string, but to make more analysis with data we need to convert it into Timestamp object
df["timeStamp"]=pd.to_datetime(df["timeStamp"]) # here we convert the column from strings to DateTime objects

df["timeStamp"].iloc[0]
# here we will create three separate new columns that show Hour, Month and Day of Week to further analysis

df["Hour"]=df["timeStamp"].apply(lambda time: time.hour)

df["Hour"] # now we have a new column showing just hour of the events
df["Month"]=df["timeStamp"].apply(lambda time: time.month )

df["Day of Week"]=df["timeStamp"].apply(lambda time: time.dayofweek)

df.head() # now we have three new separate columns that gives better understanding of the data 
# we can convert Day of Week columns from integer to their actual names for better understanding:

day={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

df["Day of Week"]=df["Day of Week"].map(day) # here we convert integers into day names 

#.map() is a method of series in pandas not a method of dataframes

df.head(10)
# here we create a countplot of the Day of Week column with the hue based off of the Reason column

plt.figure(figsize=(15,10))

sns.countplot(x="Day of Week", data=df,hue="Reason")#this represents the number of call according to days with regard to Reason

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # here we carry legend out of the plot
plt.figure(figsize=(15,10))

sns.countplot(x="Month", data=df)#this represents the number of call according to days
byMonth=df.groupby("Month").count() 

# we can make a groupby object if we want to create other types of plot that will show the number of call

byMonth
#here we create a simple plot off of the dataframe indicating the count of calls per month



plt.figure(figsize=(15,10))

plt.plot(byMonth)

plt.show()
months={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:"August",9:"September",10:"October",11:"November",12:"December"}

df["Month"]=df["Month"].map(months) # here we make month be represented by their actual names not numbers in order make it more understandable

df.head()



df["Date"]=df["timeStamp"].apply(lambda time: time.date() )

df["Date"]# here we created a new columnd that get the date from another column via a lambda expression
# here we groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.

date=df.groupby("Date").count()

date=date.reset_index()

plt.figure(figsize=(15,10))

sns.lineplot(x="Date",y="twp",data=date)
# we can create the same line plot via plt.plot() method

plt.figure(figsize=(15,10))

df.groupby("Date").count()["twp"].plot()
#here we recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call

df.groupby("Reason").count() # here we can see the number of incidents according to reason
plt.figure(figsize=(20,10)) 

df.groupby("Reason").count()["twp"].plot() # here we plot 

#For better understanding we can plot 3 different reasons separately as follows:

df[df["Reason"]=="Fire"] # here we make a condition selection where the reason is fire
# we can groupby it according the count and plot is separately as follows:

plt.figure(figsize=(15,10))

df[df["Reason"]=="Fire"].groupby("Date").count().plot() #but this shows all of the columns so we need to narrow it
plt.figure(figsize=(15,10))

df[df["Reason"]=="Fire"].groupby("Date").count()["twp"].plot(title="Fire") # so this plot shows just fire as a reason

# we copy the same code for other reasons as follows:
plt.figure(figsize=(15,10))

df[df["Reason"]=="EMS"].groupby("Date").count()["twp"].plot(title="EMS",color="green")# we just change the color and the type of reason
plt.figure(figsize=(15,10))

df[df["Reason"]=="Traffic"].groupby("Date").count()["twp"].plot(title="Traffic",color="red")
# we can make heatmaps with this data but we need not group two columns and use unstack() method in order to get a matrix data

#because sns.heatmap() accepts only matrix data

value=df.groupby(["Day of Week","Hour"]).count()["Reason"].unstack()

value
plt.figure(figsize=(15,10))

sns.heatmap(value,linecolor="black", linewidths=1)
# we can get more relational data via sns.clustermap

plt.figure(figsize=(15,10))

sns.clustermap(value,cmap="coolwarm",linecolor="black", linewidths=1)

# we can see easliy for the heat or clustermap that most of the call come between 16:00-17:00, particularly in the weekdays