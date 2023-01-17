import numpy as np
import pandas as pd 
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
data.info()
data.corr()
import matplotlib.pyplot as plt
import seaborn as sns
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.targtype1.plot(kind = 'line', color = 'g',label = 'targtype1',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.targsubtype1.plot(color = 'r',label = 'targsubtype1',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='targtype1', y='targsubtype1',alpha = 0.5,color = 'red')
plt.xlabel('targtype1')              # label = name of label
plt.ylabel('targsubtype1')
plt.title('targtype1 targsubtype1 Scatter Plot') # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.country.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.country.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
#create dictionary and look its keys and values
dictionary = {'turkey' : 'ankara','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['usa'] = "las vegas"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['usa']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
series = data['country']        # data['Defense'] = series
print(type(series))
data_frame = data[['country']]  # data[['Defense']] = data frame
print(type(data_frame))
# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['country']>200     
data[x]
# 2 - Filtering pandas with logical_and

data[np.logical_and(data['country']>200, data['iyear']>2000 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['country']>200) & (data['iyear']>2000)]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 50 :
    print('i is: ',i)
    i +=10 
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
for index,value in data[['country']][0:1].iterrows():
    print(index," : ",value)