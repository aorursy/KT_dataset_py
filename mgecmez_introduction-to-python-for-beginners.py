# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/StudentsPerformance.csv')
series = data['math score']        # data['math score'] => series
print(type(series))
dataFrame = data[['math score']]   # data[['math score']] => data frame
print(type(dataFrame))
data.info()
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
data.head()
data.columns
data.rename(columns={'gender': 'Gender', 'race/ethnicity': 'Ethnicity', 'parental level of education': 'ParentalLevelOfEducation', 'lunch': 'Lunch', 'test preparation course': 'TestPreparationCourse', 'math score': 'MathScore', 'reading score': 'ReadingScore', 'writing score': 'WritingScore'}, inplace=True)
data.columns
dfMale = data[data.Gender == "male"] # DataFrame for male
print('Male List Top 10')
print(dfMale.head(10))
print('')
dfFemale = data[data.Gender == "female"] # DataFrame for female
print('Female List Top 10')
print(dfFemale.head(10))
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
dfMale.MathScore.plot(kind='line', color='b', label='Math Score (Male)', linewidth=1, alpha=0.5, grid=True, linestyle=':')
dfFemale.MathScore.plot(color='r', label='Math Score (Female)', linewidth=1, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc='lower center')
plt.xlabel('Number of Students')
plt.ylabel('Math Scores')
plt.title('Example of Line Plot (Math Score: Male vs Female)')
plt.show()
# x = Writing Score, y = Reading Score
dfMale.plot(kind='scatter', x='WritingScore', y='ReadingScore',alpha=0.5, color='red')
plt.xlabel('Writing Score')              # label = name of label
plt.ylabel('Reading Score')
plt.title('Example of Scatter Plot (Writing vs Reading For Male)')
plt.show()
# bins = number of bar in figure
data.MathScore.plot(kind='hist', bins=100, figsize=(12,12))
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.title("Example of Histogram (Math Score)")
plt.show()
data.MathScore.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
dict = {'Turkey': 'Ankara', 'Spain': 'Barcelona', 'Italy': 'Rome', 'France': 'Paris', 'United Kingdom': 'London', 'United State': 'Washington DC'}
print("dict['Italy']: ", dict['Italy'])
dict["Spain"] = "Madrid"       # update existing entry
print(dict)
dict["Germany"] = "Berlin"     # add new entry
print(dict)
print('France' in dict)        # check include or not
del dict["Germany"]        # remove entry with key 'Germany'
print(dict)
dict.clear()               # remove all entries in dict
print(dict)

# In order to run all code you need to take comment this line
# del dict          # delete entire dict     
# print(dict)       # it gives error because dict is deleted
print(dict.keys())
print(dict.values())
print(3>2)
print(3!=2)
print(True and False)
print(True or False)
data.head()
#data = pd.read_csv('../input/StudentsPerformance.csv')
# There are only 26 students who have higher writing score than 95
flt1 = data['WritingScore'] > 95
data[flt1]
print('Filtered data count: ', len(data[flt1]))
# There are only 28 students who have higher writing score than 90 and higher reading score than 95
flt2 = np.logical_and(data['WritingScore']>90, data['ReadingScore']>95)
data[flt2]
print('Filtered data count: ', len(data[flt2]))
# This is also same with previous code line. Therefore we can also use '&' for filtering.
flt3 = (data['WritingScore']>90) & (data['ReadingScore']>95)
data[flt3]
print('Filtered data count: ', len(data[flt3]))
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
for index,value in data[['ReadingScore']][0:5].iterrows():
    print(index," : ",value)