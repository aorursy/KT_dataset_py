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
dcCharacters=pd.read_csv("../input/dc-wikia-data.csv")
marvelCharacters=pd.read_csv("../input/marvel-wikia-data.csv")
#DC Characters show 5 rows
dcCharacters.head()
dcCharacters.SEX.value_counts()
# Marvel Characters show 5 Rows
marvelCharacters.head()
# Shows both information of data
dcCharacters.info()
marvelCharacters.info()
import seaborn as sns
import matplotlib.pyplot as plt
# visualization with seaborn library - pie
indexx=marvelCharacters['ALIGN'].value_counts().index
#indexx= Bad Characters, Good Characters, Neutral Characters
valuess=marvelCharacters['ALIGN'].value_counts().values
#values= 2895, 2832,  565,    3
explode = [0,0,0]
colors = ['grey','blue','red']
plt.figure(figsize=(10,10))
plt.pie(valuess, explode=explode, labels=indexx, colors=colors, autopct='%1.1f%%')
plt.show()
valuess
indexx=dcCharacters['ALIGN'].value_counts().index
valuess=dcCharacters['ALIGN'].value_counts().values
explode = [0,0,0,0]
colors = ['grey','blue','red',"green"]
plt.figure(figsize=(10,10))
plt.pie(valuess, explode=explode, labels=indexx, colors=colors, autopct='%1.1f%%')
plt.show()
from collections import Counter
marvelCharacters['SEX'].value_counts()
characteralign=list(marvelCharacters['ALIGN'].value_counts().index)
charactersex=list(marvelCharacters['SEX'].value_counts().index)
#dictalign={"Good":0,"Bad":0,"Neutral":0}
#dictsex={"Male Characters":0,"Female Characters":0,"Agender Characters":0,"Genderfluid Characters":0}
#sumcharacter=[]
female=[]
males=[]
agenders=[]
genderfluids=[]
for each in characteralign:
    x=marvelCharacters[marvelCharacters['ALIGN']==each]
    print(x['SEX'].value_counts())
    women=len(x[x['SEX']=="Female Characters"])
    female.append(women)
    male=len(x[x['SEX']=="Male Characters"])
    males.append(male)
    agender=len(x[x['SEX']=="Agender Characters"])
    agenders.append(agender)
    genderfluid=len(x[x['SEX']=="Genderfluid Characters"])
    genderfluids.append(genderfluid)
    #women=Counter(x['SEX']=="Female Characters")

dictlist={"Male":males,"Female":female,"Agender":agenders,"GenderFluid":genderfluids}

data=pd.DataFrame(dictlist,index=characteralign)
data
characteralign2=list(marvelCharacters['ID'].value_counts().index)
charactersex2=list(marvelCharacters['SEX'].value_counts().index)
#dictalign={"Good":0,"Bad":0,"Neutral":0}
#dictsex={"Male Characters":0,"Female Characters":0,"Agender Characters":0,"Genderfluid Characters":0}
#sumcharacter=[]
female=[]
males=[]
agenders=[]
genderfluids=[]
for each in characteralign2:
    x=marvelCharacters[marvelCharacters['ID']==each]
    print(x['SEX'].value_counts())
    women=len(x[x['SEX']=="Female Characters"])
    female.append(women)
    male=len(x[x['SEX']=="Male Characters"])
    males.append(male)
    agender=len(x[x['SEX']=="Agender Characters"])
    agenders.append(agender)
    genderfluid=len(x[x['SEX']=="Genderfluid Characters"])
    genderfluids.append(genderfluid)
    #women=Counter(x['SEX']=="Female Characters")

dictlist2={"Male":males,"Female":female,"Agender":agenders,"GenderFluid":genderfluids}

data2=pd.DataFrame(dictlist2,index=characteralign2)
#data2['Toplam']=[marvelCharacters.ID.value_counts().values[0],marvelCharacters.ID.value_counts().values[1],marvelCharacters.ID.value_counts().values[2],marvelCharacters.ID.value_counts().values[3]]

data2
#visualization
import seaborn as sns
sns.jointplot(x=data.Male,y=marvelCharacters['ALIGN'].value_counts().values,data=data,kind="kde")
data.head()
data.corr()
#plt.figure(figsize=(5,5))
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)
#plt.figure(figsize=(5,5))
# Random HeatMap
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(data2.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)
listAlign=list(marvelCharacters.ALIGN.value_counts())
listAlign
plt.figure(figsize=(10,10))
sns.barplot(x=data['Male'].index,y=listAlign)
plt.xlabel("MALE Characters Align",size=12,color="red")
plt.ylabel("Count",size=12,color="red")
plt.title("Count Align of Men")
listId=list(marvelCharacters.ID.value_counts())
listId
plt.figure(figsize=(10,10))

sns.barplot(x=data2['Male'].index,y=listId)
plt.xlabel("MALE ID",size=12,color="red")
plt.ylabel("Count",size=12,color="red")
plt.title("Count Id of Man")


