# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # 
import seaborn as sns # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# read data in "BlackFriday.csv"
data = pd.read_csv("../input/BlackFriday.csv")
# get data information
data.info()
# show data columns
data.columns
# show first 10 data
data.head(10)
# correlation in consol scrren
#data.corr()

# correlation visualization square
#f,ax = plt.subplots(figsize=(10,5))
#sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#plt.show()

# correlation visualization triangle
corr = data.corr()
# Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
# Generate Color Map
colormap = sns.light_palette((210, 90, 60), input="husl")
# Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf)
# Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
# Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
# plot Purchase
# kind = kind, color = color, label = label, linewidth = width of line, alpha = opacity,
# grid = grid, linestyle = sytle of line, figsize = figure size
data.Purchase.plot(kind = 'line', color = 'g', label = 'Purchase', linewidth=1, alpha = 0.5, grid = True, linestyle = ':', figsize=(30, 10))
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('Purchase')
plt.title('Purchase Line')        # title = title of plot
plt.show()
# filtering Pandas data frame
myFilterParam = data.Purchase < 1000
data[myFilterParam].head(10)

# or
#data[ data.Purchase < 1000].head(10)
# filtering Pandas with Numpy logical and
#data[ np.logical_and( data.Purchase > 23900, data.Gender == 'F') ]

# multi filtering (i don't know how right it is?)
data[ np.logical_and( np.logical_and( data.Purchase > 23900, data.Gender == 'F'), data.Age == '18-25' ) ]
# Apply the filter to the data first, then draw the price from the filtered data
data[ data.Purchase < 400].Purchase.plot(figsize=(30, 10), kind = 'line', color = 'g', label = 'Purchase', linewidth=1, alpha = 0.8, grid = True, linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('PURCHASE')
plt.title('Purchase Line')            # title = title of plot
plt.show()
# unique values
print('Unique Product Category: {0}'.format( len( data["Product_Category_1"].unique() )))
print('Unique Product ID: {0}'.format( len( data["Product_ID"].unique() )))
print('Unique Occupation: {0}'.format( len( data["Occupation"].unique() )))
print("Maximum Purchase: {0} $".format( data.Purchase.max()))
print("Minimum Purchase: {0} $".format( data.Purchase.min()))
print("Average Purchase: {0:.2f} $".format( data.Purchase.mean()))
# {0:.2f} -> take two steps after zero
#print("Average Purchase: {0} $".format( data.Purchase.mean()))
# plot Purchase
# kind = kine, color = color, bins = number of bar in figure, figsize = figure size
data.Purchase.plot(kind = 'hist', color = "#4bd1b1", bins = 50, figsize=(12, 12))
plt.title('Purchase Histogram Graph')
plt.xlabel('Purchase')
plt.show()
# plot Scatter
# price filtering before
data[data.Purchase > 20000].plot(kind='scatter', x='Purchase', y='Product_Category_1', alpha = 0.3, color = 'b', figsize=(20, 10))
#data[data.Purchase < 500].plot(kind='scatter', x='Purchase', y='Product_Category_1', alpha = 0.3, color = 'red', figsize=(20, 10))
plt.xlabel('Purchase')
plt.ylabel('Product Category')
plt.title('Product and Purchase Scatter Plot')
plt.show()
# dictionary practice on python
myDictionary = {'hello':'merhaba',
                'pen':'kalem',
                'window':'pencere'}
print( myDictionary.keys())
print( myDictionary.values())
print('----'*20)
# add new element
myDictionary['computer'] = 'bilgisayar'
print( myDictionary)
# change element
myDictionary['hello'] = 'selam'
print( myDictionary)
# delete element
del myDictionary['hello']
print( myDictionary)
print('hello' in myDictionary)
# get 1 element value
print(myDictionary['window'])
# clear dictionary
myDictionary.clear()
# how many different cities
data["City_Category"].unique()
# find every city frequency
# P.S. -> I did this because the City_Category were string. I'm sure there are easier ways!
cityDict = {'A':0, 'B':0, 'C':0}
for each in data['City_Category']:
    cityDict[each] += 1

# draw figure
fig = plt.figure(figsize=(10,10))
# add graph
ax = fig.add_subplot(111)
# draw graph
ax.bar( cityDict.keys(), cityDict.values(), color='g')
plt.title('City Category Bar')
plt.xlabel('City Name')
plt.ylabel('Buyers')
plt.show()
# find most purchase user
print('Number of different customers: {0}\n'.format( len( data['User_ID'].unique() )))

userDict = {}
# look at both at the same time
for purch, userId in zip(data['Purchase'], data['User_ID']):
    # if already added userId in dictionary, add to purchase
    if userDict.get(userId, -1) != -1:
        userDict[userId] +=  purch
    # else add user and purchase
    else:
        userDict[userId] = purch

# find maximum purchase and buyer
print( max(zip(userDict.values(), userDict.keys() )))

#data[ data['User_ID'] == max(zip(userDict.values(), userDict.keys()))[1] ]
winnerUser = data[ data['User_ID'] == max(zip(userDict.values(), userDict.keys()))[1] ]
winnerUser.head()

#winnerUser['Purchase'].sum()

# different purchase Product
#print( len( winnerUser['Product_ID'].unique() ))
# use dictionary because age features is string
ageDict = {}
for purch, userAge in zip( data['Purchase'], data['Age']):
    if ageDict.get(userAge, -1) != -1:
        ageDict[userAge] +=  purch
    else:
        ageDict[userAge] = purch
        
# total purchase by age but no sort
print( ageDict)
print('---'*50)
# we sorted dictionary for more beatiful graph
from collections import OrderedDict
# key -> x[0]:sort dict keys, x[1]:sort dict values
age_sorted = OrderedDict( sorted( ageDict.items(), key=lambda x: x[0]))
print( age_sorted)
#print('---'*50)
#age_sorted1 = OrderedDict( sorted( ageDict.items(), key=lambda x: x[1]))
#print( age_sorted1)
# draw figure
fig = plt.figure(figsize=(10,10))
# add graph
ax = fig.add_subplot(111)
# draw graph
ax.bar( age_sorted.keys(), age_sorted.values(), color='#4f7fcc')
plt.title('Age - Purchase Bar')
plt.xlabel('Age')
plt.ylabel('Purchase')
plt.show()
# get male and female user
male_user = data[ data[ 'Gender'] == 'M'].count()[0]
female_user = data[ data[ 'Gender'] == 'F'].count()[0]
print('Male user: {0}'.format(male_user))
print('Female user: {0}'.format(female_user))
# draw figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.bar( ['Male', 'Female'], [male_user, female_user], color='#c952ad')
plt.title('Gender Bar')
plt.xlabel('Gender')
plt.ylabel('Buyers')
plt.show()
# total purchase by gender
male_purch = data[ data['Gender']=='M']['Purchase'].sum()
female_purch = data[ data['Gender']=='F']['Purchase'].sum()
print('Male user: {0}'.format(male_purch))
print('Female user: {0}'.format(female_purch))
# draw figure
fig = plt.figure(figsize=(15,10))

# graph total purchase by gender
ax1 = fig.add_subplot(121)
ax1.bar( ['Male', 'Female'], [male_purch, female_purch], color='#6152c9')
plt.title('Total Purchase by Gender')
plt.ylabel('Purchase')

# graph total purchase by gender
ax2 = fig.add_subplot(122)
ax2.bar( ['Male', 'Female'], [male_purch/male_user, female_purch/female_user], color='#d6404c')
plt.title('Average Purchase by Gender')
plt.ylabel('Purchase')

plt.show()
