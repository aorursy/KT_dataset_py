# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import collections
from operator import itemgetter  

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2015.csv")
data1 = pd.read_csv("../input/2016.csv")
data2 = pd.read_csv("../input/2017.csv")
data.info() # this code gives as abouth record number, column count, data types, memory usage.
data.shape # number of rows and columns
data.describe() # this code print statistical information abouth columns 
print("There are",data.Country.count(),"different Country")
data.Country
first10 =data.head(10)
histData = first10[['Country', 'Happiness Score']]
histData.plot(kind ='bar', x = 'Country', y= 'Happiness Score' ,title = '2015 - Top 10 Country Happiness ranking', figsize=(13,8), fontsize = 20)

first10 =data.tail(10)
histData = first10[['Country', 'Happiness Score']]
histData.plot(kind ='bar', x = 'Country', y= 'Happiness Score' ,title = '2015 - The last 10 Country Happiness ranking', figsize=(13,8), fontsize = 20)

#data.boxplot(column = 'Happiness Score', by = 'Country ')
economy_dic = {}
print('2015 - The most Rich Country and Their Happiness Score Ranking')
for key, value in sorted(data['Economy (GDP per Capita)'].items(), key = itemgetter(1), reverse = True):
    print(key, data['Country'][key], value)
new_data = data.head(10)
melted_data = pd.melt(frame = new_data, id_vars = 'Country', value_vars = ['Happiness Score','Economy (GDP per Capita)']) #melting
melted_data.pivot(index = 'Country', columns = 'variable',values='value') # reverse of melt
df = data.drop_duplicates('Region')
for i in df['Region']:
    print(i)    
count = df['Region'].count()
print("There are " , count , "different Region") # region count
    
print("The number of region country \n", data['Region'].value_counts(dropna =False))# number of region country
df.index.tolist()
index = 0
region_dic = {} # create dic for hold region and their Happiness Score sum
sorted_dic ={}
for i in df['Region']:
    i2 = -1
    count_score = 0
    for j in data['Region']:
        if (i2+1) <data['Country'].count():
            i2+=1
        if j == i:
            count_score +=data['Happiness Score'][i2]    
    region_dic[i]=count_score 
    index+=1
print("The most Happy Region")
for key, value in sorted(region_dic.items(), key = itemgetter(1), reverse = True):
    sorted_dic[key] = value
sorted_dic
plt.bar(range(len(sorted_dic)), list(sorted_dic.values()), align='center')
plt.xticks(range(len(sorted_dic)), list(sorted_dic.keys()))
plt.xticks(size = 16, rotation=90)
plt.xlabel('Regions')
plt.ylabel('Happiness Score sum')
plt.title('2015 - The most Happy Region')
plt.show()


print("2015 table")
data[['Country','Happiness Rank','Happiness Score']].head(10)
print("2016 table")
data1[['Country','Happiness Rank','Happiness Score']].head(10)
print("2017 table")
data2[['Country','Happiness.Rank','Happiness.Score']].head(10)
data2015 = data[['Country', 'Happiness Rank']] 
data2016 = data1[['Country', 'Happiness Rank']] 
data2017 = data2[['Country', 'Happiness.Rank']]
lastData= data # I will add lastData Frame 3 columns for difffent years Happiness rank
lastData['2015']= data['Happiness Rank'] # it will have 2016 2017 columns 
countDifferent =0
index1= 0
index2 = 0
bac1=[]
bac2=[]
while (index1!=lastData['Country'].count()):
    index2=0
    while(index2!=data2016['Country'].count()):
        if data2016['Country'][index2] == lastData['Country'][index1]:  
            bac1.append(data2016['Happiness Rank'][index2])
        index2+=1
    index2=0
    while(index2!=data2017['Country'].count()):
        if data2017['Country'][index2] == lastData['Country'][index1]:  
            bac2.append(data2016['Happiness Rank'][index2])
        index2+=1
    index1+=1
    
if lastData['Country'].count() != data2016['Country'].count(): # for different country len, I decided add -1, -1 means this country doesnt exist this year
    for i in range(0,lastData['Country'].count() - len(bac1)):
        bac1.append(-1)
if lastData['Country'].count() != data2017['Country'].count():
    for i in range(0,lastData['Country'].count() - len(bac2)):
        bac2.append(-1)

a = pd.Series(bac1)
b= pd.Series(bac2)
lastData['2016'] = a.values
lastData['2017'] =b.values
lastData

melted_data2 = pd.melt(frame = lastData, id_vars = 'Country', value_vars = ['2015','2016','2017'])
print(melted_data2)
melted_data2.pivot(index = 'Country', columns = 'variable',values='value') # reverse of melt