# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/books.csv')
data
#ilk bes satir

data.head()
#son bes satir

data.tail()
#data hakkinda bilgi

data.info()
data.dtypes
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
#satir sutun sayisi 

data.shape
#datada kac farkli yazar varsa diziye ekler

data.authors.unique()
data_unique = data.authors.unique()
#4664 farkli yazar var

data_unique.shape
#kac farkli dil

data.language_code.unique()
data_unique1 = data.language_code.unique()
data_unique1.shape
# turkce yazilmis kitaplar

new_language_code = data[data.language_code == 'tur']
new_language_code
x = (data['ratings_count'] > 20000)&(data['average_rating'] > 4.00)

data[x]
data[x].shape
#2014 yilindan sonra cikmis kitaplar

y = (data['original_publication_year'] > 2014)

data[y]
data1=data.loc[:,["ratings_count","ratings_3"]]

data1.plot()
data1.plot(subplots=True)

plt.show()
# Line Plot

data.ratings_count.plot(kind = 'line', color = 'g',label = 'ratings_count',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.ratings_3.plot(color = 'r',label = 'ratings_3',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper left')     # legend = etiketi plota yerlestirir

plt.xlabel('x axis')              # label = label adi

plt.ylabel('y axis')

plt.title('Line Plot')            # title = plotun basligi

plt.show()

# Scatter Plot 

# x = ratings_count, y = ratings_3

data.plot(kind='scatter', x='ratings_count', y='ratings_3',alpha = 0.5,color = 'red')

plt.xlabel('ratings_count')              # label = label adi

plt.ylabel('ratings_3')

plt.title('ratings_count ratings_3 Scatter Plot') 
# Histogram

# bins = sekildeki sutun sayisi

data.average_rating.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# threshold = esik degeri

threshold = sum(data.average_rating)/len(data.average_rating)

print('threshold = ',threshold)

data["average_rating_level"] = ["high" if i > threshold else "low" for i in data.average_rating]

data.loc[:10,["average_rating_level","average_rating","title"]] 
#NaN value/deger var mı kontrol edilir

print(data.authors.value_counts(dropna =False))
data.describe()
ndata = pd.read_csv('../input/ratings.csv')
ndata
#her kitaba okurlarin verdigi ortalama rating

ndata.groupby("book_id")[["rating"]].mean()
mdata = pd.read_csv('../input/to_read.csv')
mdata
df=data.loc[:,["authors","title"]]
df
listOfDictonaries=[]

indexMap = {}

reverseIndexMap = {}

ptr=0;

testdf = ndata

testdf=testdf[['user_id','rating']].groupby(testdf['book_id'])

for groupKey in testdf.groups.keys():

    tempDict={}



    groupDF = testdf.get_group(groupKey)

    for i in range(0,len(groupDF)):

        tempDict[groupDF.iloc[i,0]]=groupDF.iloc[i,1]

    indexMap[ptr]=groupKey

    reverseIndexMap[groupKey] = ptr

    ptr=ptr+1

    listOfDictonaries.append(tempDict)
from sklearn.feature_extraction import DictVectorizer

dictVectorizer = DictVectorizer(sparse=True)

vector = dictVectorizer.fit_transform(listOfDictonaries)
from sklearn.metrics.pairwise import cosine_similarity

pairwiseSimilarity = cosine_similarity(vector)
def printBookDetails(bookID):

    print("Title:", data[data['id']==bookID]['original_title'].values[0])

    print("Author:",data[data['id']==bookID]['authors'].values[0])

    print("Printing Book-ID:",bookID)

    print("=================++++++++++++++=========================")





def getTopRecommandations(bookID):

    row = reverseIndexMap[bookID]

    print("------INPUT BOOK--------")

    printBookDetails(bookID)

    print("-------RECOMMENDATIONS----------")

    similarBookIDs = [printBookDetails(indexMap[i]) for i in np.argsort(pairwiseSimilarity[row])[-7:-2][::-1]]
getTopRecommandations(1)
getTopRecommandations(1245)
getTopRecommandations(99)