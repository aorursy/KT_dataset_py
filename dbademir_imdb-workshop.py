# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
df = pd.read_csv("../input/tmdb_5000_movies.csv")
df.info()
df.head()
df.corr()
df.columns
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
df.budget.plot(kind = 'line', color = 'yellow',label = 'Budget',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.revenue.plot(color = 'blue',label = 'Revenue',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
# Scatter Plot 
df.plot(kind='scatter', x='budget', y='revenue',alpha = 0.7,color = 'orange')
plt.xlabel('Budget')              # label = name of label
plt.ylabel('Revenue')
plt.title('Budget Revenue Scatter Plot')
plt.show()
# Histogram
df.vote_average.plot(kind = 'hist',bins = 100,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
df.vote_count.plot(kind = 'hist',bins = 50)
plt.clf()
# Create dictionary and look its keys and values
dict = {"Mobile Phone":"iPhone", "Computer":"Lenovo"}
print(dict.keys())
print(dict.values())
# Update existing key value
dict["Computer"] = "Toshiba"
print(dict)

# Add new key-value pair
dict["Smart Watch"] = "Samsung Gear"
print(dict)

# Remove one of entries
del dict["Mobile Phone"]
print(dict)

# Check key exist or not
print('Smart Watch' in dict)

# Remove all entries
dict.clear()
print(dict)

# Deallocate dict object from cache
del dict
print(dict)
df = pd.read_csv("../input/tmdb_5000_movies.csv")
df.head()
# Define Series
series = df['budget']
print(type(series))

# Define DataFrames
data_frame = df[['revenue']]
print(type(data_frame))
print('a' == 'b')
print('c' == 1)
print(True & True)
print(3>2 or 2<1)
# Filter data using greater condition
a = df["revenue"] > 1487965080
df[a]
df[np.logical_and(df['revenue']>1487965080, df['budget']>=200000000)]
df[(df['revenue']>1487965080) & (df['budget']>=200000000)]
i=9
while i>4:
    print('value i is:', i)
    i-=1
print('Current value for i is:', i)
df_temp = df[(df['revenue']>1487965080) & (df['budget']>=200000000)]
for i in df_temp['title']:
    print('The most expensive movie is:', i)
for index, value in enumerate(df_temp['title']):
    print('Index&value of The most expensive movie is:', str(index)+'-'+value)
dict = {"Mobile Phone":"iPhone", "Computer":"Lenovo"}
for key, value in enumerate(dict):
    print('key&value of dictionary is:', str(key)+'-'+value)
# Get Index&Value of data frame
for index,value in df_temp[['title']][0:2].iterrows():
    print(index," : ",value)
a = df[df.budget>270000000]
def highest_budget():
    print(a["title"])
highest_budget()
df[df.budget>270000000]["title"]
def square_triangle(x):
    pi = 3.14
    def square():
        return x**2
    return(pi*square())
print(square_triangle(4))
def square(x):
    print(x**2)
square(4)
square = lambda x:x**2+5
print(square(4))
