# Importing the necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Read the Data of Data I-A: This dataset consists of the GSDP (Gross State Domestic Product) data for the states and union territories.

df = pd.read_csv("Downloads/ab40c054-5031-4376-b52e-9813e776f65e.csv")
df.head(10) # Read the first 10 rows across the data set to view the data 
# Let us look for the dimenstion of the datasets : 11 rows and 36 coloumns
df.shape
# Let us check the coloumn wise information with each data types
df.info()
# Get the summary of the dataframe
df.describe()
# Inspect for NAN values in the given dataframe using isnull() and sum() in the coloumn
df.isnull().sum()
# Inspect for NAN values in the given dataframe using isnull() and sum() in the row wise
df.isnull().sum(axis=1) 
# Get the percentages by dividing the sum obtained previously by the total length, multiplying it by 100 and rounding it off to
# two decimal places
round(100*(df.isnull().sum()/len(df.index)), 2)
print (df.isnull().values.any()) # there are 73 NA Values across the data frame.
print (df.isnull().values.sum())
# Use the 'drop()' function to drop the coloumn of West bengal as well as the Only states needed not Union Teritories

df=df.drop(['West Bengal1'],axis=1)
df
df.shape # Note that the one coloumn is dropped (11,36) become (11,35)---- one coloumn is dropped due to non availability of NA
# drop NA values in the rows 
#GSDP - CURRENT PRICES (` in Crore)	2016-17
#(% Growth over previous year)	2016-17
#Remove the rows: '(% Growth over the previous year)' and 'GSDP - CURRENT PRICES (` in Crore)' for the year 2016-17.

df= df[df.Duration != '2016-17'] # Entire row of 2016-17 is dropped due to NA values and no need it
df
df.tail(10)
# Inspecting again the data frame and see where still NA values present. 

round(100*(df.isnull().sum()/len(df.index)), 2)
# Get the number of retained rows using 'len()'
# Get the percentage of retained rows by dividing the current number of rows with initial number of rows

print(len(df.index))
print(len(df.index)/11)
## still we can see that around 22.22 % of data contains NAN values.
# The rows for which the sum of Null is less than five are retained
df['Himachal Pradesh'].describe()

df['Andaman & Nicobar Islands'].describe()
df['Manipur'].describe()
df['Mizoram'].describe()
df['Nagaland'].describe()
df['Punjab'].describe()
df['Rajasthan'].describe()
df['Tripura'].describe()
# We can see the minimum is only 10 and max is 104369. 25% - 14.15. Better to remove the coloum than impute it. lot of bias is there.
# Use the 'drop()' function to drop the coloumn having 22%

df=df.drop(['Himachal Pradesh','Maharashtra','Manipur','Mizoram','Nagaland','Punjab','Rajasthan','Tripura','Andaman & Nicobar Islands'],axis=1)
df
# Inspecting again the data frame and see where still NA values present. 

round(100*(df.isnull().sum()/len(df.index)), 2)
df=df.tail(3) # taken the last five rows and stored in dataframe 
df
df=df.pivot_table(index = 'Duration', aggfunc = 'mean') # using Pivot table agg. the mean and plot the graph. 
df
# Plot the graph and size the graph and legend outside the graph

import matplotlib.pyplot as plt  
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
df.plot.bar(figsize=(12,4))
plt.title("Average Growth rates of different States")
plt.xlabel('Duration')
plt.ylabel('Average Growth rates')
plt.legend(loc='best')
plt.legend(bbox_to_anchor=(1, 1), loc='lower right', ncol=5)
plt.grid(True, linewidth= 1, linestyle="--",color="g")
plt.show()




df1=df.pivot_table(index = 'Duration', aggfunc = 'mean').mean().sort_values()# Use the pivot table and did the mean calculation across the states
df1
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns
df1.plot.bar(figsize=(12,4))
plt.title(" Average Growth rates of different States over the period from 2013-2017")
plt.xlabel('Duration')
plt.ylabel('Average Growth rates')

plt.grid(True, linewidth= 1, linestyle="--",color ="r")
plt.show()

df2 = df[['All_India GDP','Tamil Nadu']] # extracting the state and the all India GDP
df2

df2.plot.bar(figsize=(4,2)) # Plot the bar graph comparing year wise my state
plt.title(" Average Growth rates of My state Tamil Nadu over the period from 2013-2017")
plt.xlabel('States')
plt.ylabel('Average Growth rates')

plt.legend(bbox_to_anchor=(1, 0), loc='upper left', ncol=2)
plt.grid(True, linewidth= 1, linestyle="--",color ="r")
plt.show()

Total_GDP =df.pivot_table(index = 'Duration', aggfunc = 'mean').sort_values(by=['2015-16'],axis=1,ascending=False) # Used Pivot to take the needed data from the dataframe and sort the highest to the lowest for 2015-16 year
Total_GDP
df3=Total_GDP.tail(-2) # The Last row is filtered and stored in df3

df3

df3.plot.bar(figsize=(15,8))
plt.title(" Total GDP of All States Across India for the Year 2015-16")
plt.xlabel('Duration')
plt.ylabel('Average Growth rates')

plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2)
plt.grid(True, linewidth= 1, linestyle="--",color ="r")
plt.show()
df3.iloc[:,0:5] # Top 5 states are given below in the data frame as well
df3.iloc[[0],[18,20,21,22,23]] # Bottom 5 states are given below in the data frame as well.
