# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Show the path you will use
data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv')
# Display content of Bitcoin data 
data.info()
# Display positive and negative correlation between columns
data.corr()
# Seaborn plot type usage on heatmap
# Annot: "to show the data in cells or not"
# Linewidth: "to customize width of the cell lines"
# Fmt: "to customize digit lenght"
# Figsize: "Figure size"
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Display the first Nth data (N=10)
data.head(10)
# Display all columns
data.columns
# Line Plot usage
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = gray square background, linestyle = sytle of line
data.Open.plot(kind='line', color='g', label='Open', linewidth=1, alpha=0.5, grid=False, linestyle="-")
data.Close.plot(color='r', label='Close', linewidth=1, alpha=0.5, grid=False, linestyle=":")
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot Sample')
plt.show()
data.columns
# Scatter Plot usage 
# x = open, y = close
data.plot(kind='scatter', x='Volume_(BTC)', y='Volume_(Currency)',alpha = 0.5,color = 'red')
plt.xlabel('BTC Volume')              # label = name of label
plt.ylabel('Currency Volume')
plt.title('BTC-Currency Scatter Plot')            # title = title of plot
# Histogram plot usage
# bins = number of bar in figure
data = data.rename(columns={'Volume_(BTC)': 'Volume_BTC'})    # We have to change column name of "Volume_(BTC)", because we need to remove parentheses to save our code from syntax error
data.Volume_BTC.plot(kind = 'hist',bins = 20,figsize = (12,12))
plt.show()

#Not a good example but it shows how to save our code from problematic column name usages 
# clf() = cleans it up again you can start a fresh
data.Open.plot(kind='hist', bins=50)
#plt.clf()
# We cannot see the plot due to clf() if you remove comment mark "#"
#create dictionary and look its keys and values
#faster than lists thats why we use dictionaries
dictionary={'Spain':'Madrid','Turkey':'Istanbul'}
print(dictionary)
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['Spain']="Seville" # Change the value for Spain
dictionary['Turkey']="Ankara" # Change the value for Turkey
print(dictionary)

del dictionary['Turkey'] # Delete "Turkey" key and its value from dictionary
print(dictionary)

dictionary['Germany']="Berlin" # Add "Germany" as a new key and "Berlin" as a new value to dictionary
print(dictionary)

print('Germany' in dictionary)  # Return true if "Germany" exists in dictionary or false

# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data=pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv')
series=data['Open']   # a serie (1 dimenson)
#print(series)
print(type(series))

data_frame=data[['Open']]  # a data frame (2 dimension)
#print(data_frame)
print(type(data_frame))
# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators (return True or False)
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['Weighted_Price']>2000     # Display prices greater than $2000
data[x]
# 2 - Filtering pandas with logical_and (&)
data[np.logical_and(data['Weighted_Price']>200, data['Low']>100 )]

#or you can use
#data[(data['Weighted_Price']>200) & (data['Low']>100)]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['High']][0:1].iterrows():
    print(index," : ",value)