# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import time

import datetime

from datetime import datetime

import collections



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/earthquake.csv')

data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
a=data.loc[:,"date"]                            

b=data.loc[:,"time"]

print (a[0], b[0])

type(a)                                        



temp = a+"_"+b                               # this is the combined form we would like to achieve

timeformat="%Y.%m.%d_%H:%M:%S %p"



new=[datetime.strptime(x, timeformat) for x in temp]



#for i in temp:

#    i=datetime.strptime(i,timeformat)

#why not this way??



print("temp =",type(temp),"\n""new =",type(new),"\n""data.time =",type(data.date))



data.time=new



data.rename(columns={'time': 'newtime'}, inplace=True) 

del data["date"]                            # we dont need it anymore as all stored in date.time
print(data.iloc[:5,1])

data.head(5)
data["year"]=[int(datetime.strftime(x,"%Y")) for x in data.newtime]

data["month"]=[int(datetime.strftime(x,"%m"))+int(datetime.strftime(x,"%Y"))*12 for x in data.newtime]

data.info()

data.head(2)
tur=data.country == "turkey"

real=data.richter > 1



cit=data[tur & real].city

cits=cit.unique()



print("Total Cities =",cit.size)



a=0

for i in cits:

    a=a+1

    if a==len(cits):

        print("Unique Cities = {}".format(a))



f=Counter(cit)

newf=f.most_common()



print(type(f))

print(type(newf))
print("So in the table above we can see that, in given date range, {} eq happened in {} cities" .format(cit.size, a))
maks=max(f, key=f.get) 

most=f.most_common(5)[0]

most2=f.most_common(5)[1]

print("Max number of eq occured in {} with {} eq and second is {} with {}" .format(maks.upper(),most[1],most2[0].upper(),most2[1]))
data.head(5)
def dist(baslik):

    

    

    tur = data.country=="turkey"                # There arent many records before 2000 roughly, so lets filter after 1998, also just take magnitudes over 2 

    richter = data.richter > 2

    yearfilter = data.year > 1998

    md = data.md > 2

    

    datatr= data[tur & richter & yearfilter & md]

    

    plt.figure(figsize=(10,5))

    plt.hist(datatr[baslik], bins=30, color="blue")

    plt.ylabel("Frequency")

    plt.title(baslik)



ozet=["richter", "year", "md", "xm","lat","long"]

    

for each in ozet:

    dist(each)

    
yearfilter = data.year > 1998

data[yearfilter][["year","richter"]].groupby(["year"], as_index = False).mean().sort_values(by = "richter", ascending = False)
yearfilter = data.year > 1998

data[yearfilter][["year","richter"]].groupby(["year"], as_index = False).count().sort_values(by = "richter", ascending = False)
data.columns[data.isnull().any()]  
data.isnull().sum()