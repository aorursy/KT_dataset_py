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

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/movies_metadata.csv')
data.info()
data.columns
data.head(10)
data.dtypes #column types
data.describe()
f1= (data.original_language == 'tr')
data1 = data[f1]
data1.head(10)
#data_view=data1.loc[:,['original_language','original_title','overview','vote_count','vote_average','popularity','release_date','runtime','revenue']].sort_values('release_date',ascending=False)

data_view=data1.loc[:,['original_language','original_title','overview','vote_count','vote_average','popularity','release_date','runtime','revenue']].sort_values('release_date',ascending=False)
data_view

data.corr()
#correlation map
#featurelar arasındaki ilişkiyi görmek için kullanılır
#1 e yakın oldukça ilişkili
#-1 e yakınlaştıkça ters orantılı

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data_view.vote_count.plot(color='r',label = 'Vote Count',linewidth=1,alpha = 1,grid = True,linestyle = ':')
data_view.revenue.plot(color='g', label = 'Revenue',linewidth=1,alpha=1,grid=True,linestyle='-.')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = vote_count, y = vote_average
data.plot(kind='scatter', x='vote_count', y='vote_average',alpha = 0.5,color = 'red')
plt.xlabel('vote_count')              # label = name of label
plt.ylabel('vote_average')
plt.title('vote_count vote_average Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.vote_count.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
#dictionary examples
dictionary = data_view.set_index('original_title')['vote_count'].to_dict()
dictionary
dictionary['Kedi'] = 50.0

#del dictionary['Siccin 3: Cürmü Ask']
print('Pek Yakında' in dictionary)
dictionary.clear()
#del dictionary
dictionary
i=0
for index,value in data_view[['original_title']][::1].iterrows():
    print(index," : ",value)
    i = i+1
print("Toplam: ",i)
#user definition function
def factorial(x):
    i=1
    fac=i
    while i<=x:
        fac = fac*i
        i = i+1
    return fac

factorial(3)

    
#lambda function

lis = [1,2,3,4,5]
lis2=[]

x = lambda i:i**2

for i in lis:
    lis2.append(x(i))
    
lis2

#map,zip,filter

lis3 = map(factorial,lis)

for i in lis3:
    print(i)

lis4 = [1,2,3,4,5]
lis5 = ['a','b','c','d','e']
lis6 = [10,20,30,40,50]


data1 = zip(lis4,lis5)
unlis1,unlis2 = zip(*data1)
print(unlis1)
print(unlis2)

data2 = map(lambda x,y:x*y,lis4,lis6)
data3 = filter(lambda x: x%2==0,lis4)

for i in data1:
    print(i)

for i in data2:
    print(i)

print(list(data3))

#list comprehention

lis7 = [100,50,250,40,15,312,475,86,20,500]
lis7_state = ['large' if i>100 else 'medium' if i<=100 and i>=50 else 'small' for i in lis7]
print(lis7_state)


total_vote_average = data.vote_average.mean()
print(total_vote_average)

data_view=data.loc[:,['original_language','original_title','overview','vote_count','vote_average','popularity','release_date','runtime','revenue']].sort_values('release_date',ascending=False)
f1= (data_view.original_language == 'tr')
data_view = data_view[f1]

data_view['state'] = ['good' if each > total_vote_average else 'bad' for each in data_view.vote_average]
data_view





