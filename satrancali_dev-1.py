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
# I used " encoding " because utf-8 encoding show me error message
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv" , encoding="ISO-8859-1")
data.info()

# I listed columns names because , this database has got a lot of column names, 
# so print(data.columns) not enough for all list columns, now I can see all of them
for i in data.columns:
    print(i)

# I want to see some examples

data.tail()

# I selected some important columns; iyear, imonth, country_txt, region_txt, attacktype1_txt, targtype1_txt , success
# I want to use corr method and this method for integer values so country_txt == country because country column is integer, region_txt = region
# attacktype1_txt = attacktype1 , targtype1 = targtype1_txt
# these are integer values iyear,imonth,country,region,attacktype1,targtype1,success
# I want to delete other columns so I make a filter
filter1 = ["iyear","imonth","country","region","attacktype1","targtype1","success"]
#list_filter = []
#for each in data:
 #   if(each in filter1):
  #      pass
   # else:
    #    list_filter.append(each)
#datam = data.drop(list_filter, axis=1)
f,ax = plt.subplots(figsize=(70, 70))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

# This values are not enough for using so I looked all columns 
rows, cols = data.shape
flds = list(data.select_dtypes(include=['number']).columns)
#corr = data.corr().values


#data.select_dtypes(include=['number']).columns

#print(data['imonth'].corr(data['iyear']))
dictionary_i=[]
for i in flds:
    for a in flds:
        if(0.7 < data[a].corr(data[i])):
            if(i!=a):
                print(i,a)
                dictionary_i.append(i)

#Now I can see that What I want to see.
#print(dictionary_i)
# data.filter(items=['eventid', 'iyear', 'crit3', 'alternative', 'targtype1', 'targsubtype1', 'targtype2', 'targtype2', 'targsubtype2', 'targsubtype2', 'natlty2', 'targtype3', 'targsubtype3', 'claimmode2', 'claimmode3', 'weaptype1', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'weapsubtype4', 'weapsubtype4', 'weapsubtype4', 'weapsubtype4', 'weapsubtype4', 'weapsubtype4', 'nkillus', 'nwound', 'ndays', 'ransomamt', 'nreleased', 'INT_LOG', 'INT_LOG', 'INT_IDEO', 'INT_IDEO', 'INT_ANY', 'INT_ANY'])
#Columns have got a lot of NaN values, I should delete
#new data.filter(items=['eventid', 'iyear', 'crit3', 'targtype1', 'targsubtype1', 'weaptype1', 'INT_LOG', 'INT_LOG', 'INT_IDEO', 'INT_IDEO', 'INT_ANY', 'INT_ANY'])
#Now I should find text of these integer values.
#I want to change integer values to text values
# targtype1_txt targsubtype1_txt weaptype1_txt
#data.filter(items=['eventid', 'iyear', 'crit3', 'targtype1_txt', 'targsubtype1_txt', 'weaptype1_txt', 'INT_LOG', 'INT_LOG', 'INT_IDEO', 'INT_IDEO', 'INT_ANY', 'INT_ANY'])
# I think I should delete these, 'eventid', 'INT_LOG', 'INT_LOG', 'INT_IDEO', 'INT_IDEO', 'INT_ANY', 'INT_ANY'
data.filter(items=['iyear', 'crit3', 'targtype1_txt', 'targsubtype1_txt', 'weaptype1_txt'])

data.iyear.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
#I can see last years criminal upper more than before
data.targtype1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# I can see number 14 is most popular targtype1
#I want to look targtype1_txt for looking name
data.filter(items=['targtype1_txt', 'targtype1'])
# I see  targtype1 14 is Private Citizens & Property
#I want to look others
data.targsubtype1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data.filter(items=['targsubtype1_txt', 'targsubtype1'])
#I want to look others
data.weaptype1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data.filter(items=['weaptype1_txt', 'weaptype1'])
data.plot(kind='scatter', x='iyear', y='targtype1',alpha = 0.01,color = 'red')
plt.xlabel('iyear')              # label = name of label
plt.ylabel('targtype1')
plt.title('targtype1 values compare to year')
data.plot(kind='scatter', x='iyear', y='targsubtype1',alpha = 0.01,color = 'red')
plt.xlabel('iyear')              # label = name of label
plt.ylabel('targsubtype1')
plt.title('targsubtype1 values compare to year')
data.plot(kind='scatter', x='iyear', y='weaptype1',alpha = 0.01,color = 'red')
plt.xlabel('iyear')              # label = name of label
plt.ylabel('weaptype1')
plt.title('weaptype1 values compare to year')
