# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/creditcard.csv")
df.info()
df.head()
df.corr()
df.columns
#correlation map
f,ax = plt.subplots(figsize=(21,21))
sns.heatmap(df.corr(), annot=True, linewidths=.6, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
df.Time.plot(kind = 'line', color = 'red',label = 'Time',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.Amount.plot(color = 'blue',label = 'Amount',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
# Scatter Plot 
df.plot(kind='scatter', x='Time', y='Amount',alpha = 0.7,color = 'blue')
plt.xlabel('Time')          
plt.ylabel('Amount')
plt.title('Time Amount Scatter Plot')
plt.show()
# Histogram
df.Time.plot(kind = 'hist',bins = 75,figsize = (10,10))
plt.show()
# clf() = cleans it up again you can start a fresh
df.Time.plot(kind = 'hist',bins = 50)
plt.clf()
# Create dictionary and look its keys and values
dict = {"Volkswagen":"Passat", "Renault":"Megane"}
print(dict.keys())
print(dict.values())
# Update existing key value
dict["Volkswagen"] = "Golf"
print(dict)

# Add new key-value pair
dict["Nissan"] = "Micra"
print(dict)

# Remove one of entries
del dict["Renault"]
print(dict)

# Check key exist or not
print('Nissan' in dict)

# Remove all entries
dict.clear()
print(dict)

# Deallocate dict object from cache
del dict
print(dict)
df = pd.read_csv("../input/creditcard.csv")
df.head()
# Define Series
series = df['Time']
print(type(series))

# Define DataFrames
data_frame = df[['Amount']]
print(type(data_frame))
print('A' == 'a')
print(0 == 'zero')
print(5 != 2)
print(False & False)
print(1<2 or 4>3)
# Filter data using greater condition
a = df["Amount"] > 12910.93
df[a]
df[np.logical_and(df['Time']>48401.0, df['Amount']>=19656.53)]
df[(df['Time']>48401.0) & (df['Amount']>=19656.53)]
i=-3
while i<5:
    print('value i is:', i)
    i+=1
print('Current value for i is:', i)
dict = {"Volkswagen":"Passat", "Renault":"Megane"}

for key, value in enumerate(dict):
    print('key&value of dictionary is:', str(key)+'-'+value)
# Get Index&Value of data frame
for index,value in df[['V1']][0:2].iterrows():
    print(index," : ",value)
a = df[df.Time>0.117396]
def min_time():
    print(a["Amount"])
min_time()
df[df.Time>0.117396]["Amount"]
def circle_circumference(x):
    pi = 3.14
    return 2*pi*x
print(circle_circumference(3))
def square(x):
    print(x**2)
square(6)
square = lambda x:x**3+4
print(square(6))
