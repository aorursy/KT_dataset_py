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
data=pd.read_csv("../input/StudentsPerformance.csv")

data.info()
data.columns
data.columns=[i.split()[0]+"_"+i.split()[1]  if len(i.split())>1 else i for i in data.columns]
data.head(10)
data.tail(10)
data.rename(columns={"race/ethnicity":"group","gender":"sex"},inplace=True)



data.group.unique()
data.corr()
f,ax = plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(),annot=True)

plt.show()
f=0

m=0

for i in data.sex:

    if i=="female":

        f+=1

    elif i=="male":

        m+=1

list_sex={"female":f,"male":m}

print(list_sex)
data.head(10)

data.describe()

data.plot(kind="hist",subplots=True,grid=True)

plt.show()
data[(data["math_score"]>90) & (data["reading_score"]>90) & (data["writing_score"]>90)]
data_groupby=data.groupby("sex")

data_groupby.mean()
data_group=data.groupby("group")

data_group.mean()
data_parental=data.groupby("parental_level")

data_parental.mean()
def total_score(math_score,reading_score,writing_score):

    mathList=[]

    for key,value in list(math_score.iteritems()):

        mathList.append(value)

    readList=[]

    for key,value in list(reading_score.iteritems()):

        readList.append(value)

    writeList=[]

    for key,value in list(writing_score.iteritems()):

        writeList.append(value)

    mathreadwrite=zip(mathList,readList,writeList)

    totalScore=[]

    for math,read,write in mathreadwrite:

        totalScore.append(math+read+write)

    return totalScore



data["total_score"]=total_score(data["math_score"],data["reading_score"],data["writing_score"])





        

    

data.sort_values("total_score",ascending=False,inplace=True)
data_parental=data.groupby("parental_level")

data_parental.mean().sort_values("total_score",ascending=False)