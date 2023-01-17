import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Let's read data

data = pd.read_csv('../input/fifa19/data.csv')
# To get information about this data

data.info()
# What are the columns of this data ?

data.columns
# Look at top 5 data :

data.head()
# Corrolation

data.corr()
#correlation map

f,ax = plt.subplots(figsize=(50, 50))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

# to see big picture double-click
data["Special"].corr(data["BallControl"])

# https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials
# Line Plot :



data.Age.plot(kind='line',color='g',Label='Age',linewidth=1,alpha=0.5,grid=True,linestyle=':')

data.Overall.plot(color='b',Label='Value',linewidth=1,alpha=0.5,grid=True,linestyle='-')



plt.legend(loc='upper right')

plt.xlabel('Age')

plt.ylabel('Overall')

plt.title('Age - Value Analysis')

plt.show()
# Scatter Plot :



# We will use columns of 'BallControl' and 'Special'

# But some rows are NULL. So we will create a new data with filter



filter1 = data["BallControl"]>0

data2=data[filter1]

data2.info()
data2.plot(kind="scatter",y="Special",x="BallControl",alpha=0.5, color="g")

plt.ylabel("Special")

plt.xlabel("Ball Control")

plt.title("Special - Ball Control")

plt.show()
# Histogram Plot :



data2.Age.plot(kind="hist", bins=50)

plt.show()

# It shows age / count

# bins = number of bar in figure
#create dictionary and look its keys and values

dictionary1 = {"Turkey":"Istanbul","UK":"London","Germany":"Berlin"}



keys1 = dictionary1.keys()

values1 = dictionary1.values()



print("Keys :")

print(keys1)

print(type(keys1))



print()

print("Values :")

print(values1)

print(type(values1))

# Keys must be unique

dictionary1["Turkey"] = "Ankara" # update existing entry

print(dictionary1)



dictionary1["France"] = "Paris" # add new entry

print(dictionary1)



del dictionary1["UK"]

print(dictionary1)



print("Germany" in dictionary1) # check include or not

print("UK" in dictionary1) # check include or not



dictionary1.clear() # remove all entries in dict

print(dictionary1)
# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary1)       # it gives error because dictionary is deleted
# Read Data :



data3 = pd.read_csv("../input/fifa19/data.csv")

data3.info()
series_Age = data3["Age"] # --> This is SERIE : only one []

print("Type of Series : ", type(series_Age))

print(series_Age)



print()

data_frame1 = data3[["Age"]] #--> This is DATA FRAME  : two []

print("Type of data_frame : " , type(data_frame1))

print(data_frame1)
data_frame2 = data3[["Age","Overall","Potential"]]

print(data_frame2)
filter1 = data3["Age"] > 43

data3[filter1]
filter2 = data3["Overall"] < 71

data3[filter1 & filter2]
data3[np.logical_and(filter1,filter2)]
i = 0

while i != 4:

    i = i+1

    print(i)
lis = [5,8,7,6,1]

for i in lis:

    print('i --> ' ,i)

    

print()

for index,value in enumerate(lis):

    print(index , ' : ' , value)

    

print()

dictionary2 = {"Turkey" : "Ankara", "UK" : "London" , "Germany" : "Berlin"}

for key,value in dictionary2.items():

    print("Capital of " , key , " is " , value)
# get from 0 to 1 rows 

for index,value in data3[["Age"]][0:1].iterrows():

    print(index , ' :'  , value)
# also we can get between 5th and 6th rows : 

for index , value in data3[["Age"]][5:7].iterrows():

    print(index , ' : ' , value)