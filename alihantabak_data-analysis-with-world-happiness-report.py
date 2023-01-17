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
data2015 = pd.read_csv('../input/2015.csv') #Read our datas from csv files.
data2016 = pd.read_csv('../input/2016.csv')
data2017 = pd.read_csv('../input/2017.csv')
data2015.info()
data2015.columns
data2015.shape #That means we've got 158 rows and 13 columns
data2015.corr() #give us to correloation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data2015.corr(), annot=True,fmt='.1f',ax=ax) #this give us a figure of correlation map
plt.show() #dismiss an information column
data2015.head(10) #Default value is 5
data2015.columns
data2015.Freedom.plot(kind='line',color='g',label='Freedom',linewidth=1,alpha=0.5,linestyle='-.')
data2015.Family.plot(kind='line',color='red',label='Family',linewidth=1,alpha=0.5,linestyle=':')
plt.legend()
plt.xlabel='x axis'
plt.ylabel='y axis'
plt.show()
data2015.Freedom.plot(kind='line',color='g',label='2015Freedom',linewidth=1,alpha=0.5,linestyle='-.')
data2016.Freedom.plot(kind='line',color='red',label='2016Freedom',linewidth=1,alpha=0.5,linestyle=':')
plt.legend()
plt.xlabel='x axis'
plt.ylabel='y axis'
plt.show()
data2015.columns
data2015.plot(kind='scatter',x='Family',y='Generosity',alpha=0.5,color='green')
plt.xlabel = 'Family'
plt.ylabel = 'Generosity'
plt.show()
#histogram plot
data2015.Generosity.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data2015
#Filtering
filtered_data = data2015['Freedom'] > 0.65
data2015[filtered_data]
#Filtering with use logical_and method
data2015[np.logical_and(data2015['Freedom']>0.65, data2015['Family']>1.34)]
#Let's use for loop on this project
for index,value in data2015[['Family']][0:2].iterrows():
    print("index: ",index,"value: ",value)
#List comprehension usage
average = sum(data2015.Family)/len(data2015.Family)
data2015["Family_Situtation"] = ["Above Average" if i>average else "Average" if i==average else "Below Average" for i in data2015.Family]
data2015.columns #We'll see Family_Situtation is in our columns right now!
data2015.head()
#Use a filter for see the Below Average countries
filtered_data = data2015.Family_Situtation =="Below Average"
data2015[filtered_data]
print(data2015['Region'].value_counts(dropna=False))
#That means; there are 40 Sab-Saharan Africa, 29 Central and Eastern Europe etc. countries in this report.
data2015.describe()
#This method gives us just numerical values.
#min is minimum value of a feature
#max is maximum value of a feature
#%50 is median of a feature
#%25 is median of %50 and min values in a feature (that named as Q1)
#%75 is median of ½50 and max values in a feature (that named as Q3)

#Outlier data: the value that is considerably higher or lower from rest of the data
#IQR = (Q3-Q1)
#If a value smaller than Q1-(1.5xIQR) or bigger than Q3+(1.5xIQR) that means; that data is an outlier data!
#boxplot shows us min, max, quantiles(%25,%50,%75)
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There is an outlier data. It's shape a circle and it located on 'Below Average'
data2015.boxplot(column='Happiness Score',by='Family_Situtation')
plt.show()
#create a small data from data2015
smaller_data = data2015.tail()
smaller_data
#melt method is a method of pandas
#frame shows us which data will used
#id_vars show us which feature won't melt
#value_vars show us which features will melt
melted_data = pd.melt(frame=smaller_data,id_vars='Country',value_vars=['Generosity','Family'])
melted_data
#pivot method is used for reverse of melting.
melted_data.pivot(index='Country',columns='variable',values='value')
#concatenate datas
data1 = data2015.head()
data2 = data2015.tail()
concat_data = pd.concat([data1,data2],axis=0,ignore_index=False)
#'[data1,data2]' means concat data1 and data2
#'axis=0' means adds dataframes in row
#'ignore_index = True' means give new indexes our datas. Firstly, Rwanda's index was 153 but we ignore it and its index is 5 now.
concat_data
#adds dataframes in columns
data1 = data2015['Family'].head()
data2 =data2015['Generosity'].head()
concat1_data = pd.concat([data1,data2],axis=1)
concat1_data
#if we want to learn our datas' type, we use dtypes method.
data2015.dtypes
#we can convert types of features. We use astype method for it.
#for example; type of Freedom feature is float and I want to convert it to int.
data2015['Freedom'] = data2015['Freedom'].astype('int')
data2015.dtypes
#as you can see; type od Freedom data is changed.
data2015.head()
#All of the indexes of Freedom are 0 right now. Because we converted it and 0.xxx is 0 right now.
#for this reason we should read the data again. This is just an example.
data2015 = pd.read_csv('../input/2015.csv')
data2015.head()
#let's add our Family_Situtations feature quickly again.
average = sum(data2015.Family)/len(data2015.Family)
data2015["Family_Situtation"] = ["Above Average" if i>average else "Average" if i==average else "Below Average" for i in data2015.Family]
data2015.columns
#we added it!
#Some datas could be have some NaN (Not-a-Number) values. That datas named as 'Missing Value' Let's check it!
data2015.info()
#Check it with a feature
data2015['Region'].value_counts(dropna=False)
assert data2015.Freedom.dtype==float #that returns nothing. Because that's true.