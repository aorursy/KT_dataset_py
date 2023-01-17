# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from math import sin, cos, sqrt, atan2, radians
import datetime as dt
import math
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
filePath = "../input/DBLP-citation-Jan8.csv"
df = pd.read_csv(filePath)
df.info()
print("before: ",len(df))
df = df[pd.notnull(df['paperTitle'])]
print("after: ",len(df))
df.head()
paperToReferenceCountMap = dict()
def paperToReferenceCount(row):
    paperTitle=row["paperTitle"]
    reference_count=row["reference_count"]
    if(paperTitle not in paperToReferenceCountMap):
        paperToReferenceCountMap[paperTitle]=reference_count

df.apply(lambda row: paperToReferenceCount(row), axis=1)   
import json
authorToPaperMap={}
def createData(row):
    authorJson =json.loads(row['author'])
    paperTitle=row["paperTitle"]
    for key,authorName in authorJson.items():
        if authorName not in authorToPaperMap:
            authorToPaperMap[authorName]=[]
        authorToPaperMap[authorName].append(paperToReferenceCountMap[paperTitle])
    
df.apply(lambda row : createData(row), axis=1)

def getHIndex(citations):
    maxi = 0
    for i in range(0,len(citations)):
        hindex=0;
        if(citations[i] >= len(citations)-i):
            hindex=len(citations)-i;
        if(maxi < hindex):
            maxi=hindex;
    return maxi
authorHIndex = {}
for author, papers in authorToPaperMap.items():
    sortedPapers=sorted(papers)
    authorToPaperMap[author]=sortedPapers
    if(author not in authorHIndex):
        if(sortedPapers[len(sortedPapers)-1]==0):
            authorHIndex[author]=0
        else:            
            authorHIndex[author]=getHIndex(sortedPapers)

sorted_by_value = sorted(authorHIndex.items(), key=lambda kv: kv[1],reverse=True)
sorted_by_value[0:100]
hIndexMap = {}
for name,hindex in authorHIndex.items():
    
    if hindex not in hIndexMap:
        hIndexMap[hindex]=0
    hIndexMap[hindex]+=1
    
hIndexDf= pd.DataFrame(list(hIndexMap.items()),
                      columns=['h-index', 'value'])
df.head()
authorToYearMap = {}

def findAuthorYear(row):
    authorJson =json.loads(row['author'])
    year=row["year"]
    for key,authorName in authorJson.items():
        if authorName not in authorToYearMap:
            authorToYearMap[authorName]=year
        if(year < authorToYearMap[authorName]):
            authorToYearMap[authorName]=year
            
df.apply(lambda row : findAuthorYear(row), axis=1)
authorToYearMap
authorToAgeMap  ={}
maxi = 0
for author,year in authorToYearMap.items():
    authorToAgeMap[author] = 2012 - year
authorHIndexDF = pd.DataFrame(list(authorHIndex.items()),
                      columns=['AuthorName', 'h-index'])
def getAuthorCareerAge(row):
    author = row["AuthorName"]
    return authorToAgeMap[author]

authorHIndexDF["CareerAge"] = authorHIndexDF.apply(lambda row : getAuthorCareerAge(row), axis=1)
authorHIndexDF.info()
# import matplotlib.pyplot as plot
# f, ax = plot.subplots(1,1, figsize=(30,12))
# sns.countplot(authorHIndexDF["CareerAge"],order = authorHIndexDF["CareerAge"].value_counts().iloc[:50].index)
ageCountMap = {}
for auth, age in authorToAgeMap.items():
    if(age > 40):
        continue
    if age not in ageCountMap:
        ageCountMap[age]=0
    ageCountMap[age]+=1
ageCountMapDF = pd.DataFrame(list(ageCountMap.items()),
                      columns=['age', 'countOfPapers'])
# sns.countplot(x=ageCountMapDF['age'], y=ageCountMapDF['countOfPapers'])
ageCountMapDF.plot(x='age',y='countOfPapers')
authorHIndexAgeGroupedDF.plot()
authorHIndexDF
# authorHIndexDF.head()
import seaborn as sns
authorHIndexDF.plot(x=authorHIndexDF["CareerAge"],y=authorHIndexDF["h-index"])


