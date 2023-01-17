# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Import the data into the environment 
Data=pd.read_csv("../input/traffic-collision-data-from-2010-to-present.csv")

#Checking the size of the data available
length = Data.shape

print("Number of samples present in the data are %d" %(length[0] ))
print("Number of fields present in the data is %d" %(length[1] ))
print("Fields present in the data are as follows:")
print(list(Data))

Data['Year'] = pd.DatetimeIndex(Data['Date Occurred']).year
Data['Month'] = pd.DatetimeIndex(Data['Date Occurred']).month
Data['Day'] = pd.DatetimeIndex(Data['Date Occurred']).day
Data['Week'] = pd.DatetimeIndex(Data['Date Occurred']).week
Data['Weekday'] = pd.DatetimeIndex(Data['Date Occurred']).weekday

def hour(x):
    x=x//100
    return (x)

Data["Hour"]=Data["Time Occurred"].apply(hour)



Data.head()
# Remove any data recorded for the year 2019. Since the data is recorded for only four days in 2019
Data=Data[Data["Year"] != 2019]
Data=Data.iloc[:,4:]


plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data["Year"],bins=range(2010,2020),width=0.8,align='left')
for i,j in zip(Data["Year"].unique(),n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-35,5), textcoords='offset points',fontsize=14)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Year")
plt.ylabel("Number of Collisions")
plt.xticks(range(2010,2019))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
#Analyze the trends along different months each year
Data_2018=Data[Data["Year"] == 2018]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2018[""],bins=range(1,14),width=0.8,align='left')
for i,j in zip(Data_2018["Month"].unique(),n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-35,5), textcoords='offset points',fontsize=14)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Month")
plt.ylabel("Number of Collisions")
plt.xticks(range(1,13))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()

###f, axarr = plt.subplots(2,5,figsize=(20,10))
##for i in range (0, 5): 
  ##  axarr[0, i].hist(Monthwisedata[i]["Month"],bins=range(1,14),width=0.8,align='left')
    #axarr[0, i].set_title(year[i])
    #axarr[0, i].set_xlabel('Month')
    #axarr[0, i].set_ylabel('Number of Collisons')
    #axarr[0, i].set_xticks(range(1,13))
    
#for i in range (0, 4): 
 #   axarr[1,i].hist(Monthwisedata[i+5]["Month"],bins=range(1,14),width=0.8,align='left')
  #  axarr[1,i].set_title(year[i+5])
   ##axarr[1,i].set_ylabel('Number of Collisons')
    #axarr[1,i].set_xticks(range(1,13))
    
#axarr[1,4].set_visible(False)###


Data_2017=Data[Data["Year"] == 2017]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2017["Month"],bins=range(1,14),width=0.8,align='left')
k=Data_2017["Month"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-35,5), textcoords='offset points',fontsize=14)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Month")
plt.ylabel("Number of Collisions")
plt.xticks(range(1,13))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()

Data_2016=Data[Data["Year"] == 2016]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2016["Month"],bins=range(1,14),width=0.8,align='left')
k=Data_2016["Month"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-35,5), textcoords='offset points',fontsize=14)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Month")
plt.ylabel("Number of Collisions")
plt.xticks(range(1,13))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
Data_2018=Data[Data["Year"] == 2018]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2018["Hour"],bins=range(0,25),width=0.8,align='left')
k=Data_2018["Hour"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-15,5), textcoords='offset points',fontsize=10)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Hour")
plt.ylabel("Number of Collisions")
plt.xticks(range(0,24))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
Data_2017=Data[Data["Year"] == 2017]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2017["Hour"],bins=range(0,25),width=0.8,align='left')
k=Data_2017["Hour"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-15,5), textcoords='offset points',fontsize=10)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Hour")
plt.ylabel("Number of Collisions")
plt.xticks(range(0,24))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
Data_2016=Data[Data["Year"] == 2016]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2016["Hour"],bins=range(0,25),width=0.8,align='left')
k=Data_2016["Hour"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-15,5), textcoords='offset points',fontsize=10)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Hour")
plt.ylabel("Number of Collisions")
plt.xticks(range(0,24))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
Data_2018=Data[Data["Year"] == 2018]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2018["Weekday"],bins=range(0,8),width=0.8,align='left')
k=Data_2018["Weekday"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-15,5), textcoords='offset points',fontsize=10)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Hour")
plt.ylabel("Number of Collisions")
plt.xticks(range(0,7))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
Data_2017=Data[Data["Year"] == 2017]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2017["Weekday"],bins=range(0,8),width=0.8,align='left')
k=Data_2017["Weekday"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-15,5), textcoords='offset points',fontsize=10)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Hour")
plt.ylabel("Number of Collisions")
plt.xticks(range(0,7))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
Data_2016=Data[Data["Year"] == 2016]
plt.figure(figsize=(15,10))
(n,bins, width) = plt.hist(Data_2016["Weekday"],bins=range(0,8),width=0.8,align='left')
k=Data_2016["Weekday"].unique()
k.sort()
for i,j in zip(k,n):
    plt.annotate('%s' %j, xy=(i,j), xytext=(-15,5), textcoords='offset points',fontsize=10)
    #plt.annotate('(%s,' %i, xy=(i,j))
plt.xlabel("Hour")
plt.ylabel("Number of Collisions")
plt.xticks(range(0,7))
plt.title("Number of collisons occured each year in the period beginning 2010 to end of 2018")
plt.show()
