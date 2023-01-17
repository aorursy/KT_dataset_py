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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display 
df = pd.read_csv('../input/athlete_events.csv').fillna(0).head(1000)
#display(df)
nameOfCountry = df['Team'].unique()
nameOfYear = df['Year']
#print(nameOfCountry,type(nameOfCountry))
#for each in nameOfYear:
#dfTeam= pd.DataFrame(df['NOC'])
#dfYear = pd.DataFrame(df['Year'])
#dfCombined = pd.concat([dfTeam,dfYear],axis=1).values# Convert the dataframe to nump array
df['TeamWithYear'] = df[['Team','Year']].apply(tuple,axis=1)
#display(df['TeamWithYear'])
newList=[]
dictCountry=dict()
for each in nameOfCountry:#China is taken
    for elem in df['TeamWithYear']:
        if each == elem[0]:
            newList.append(elem)
    #print('newList: ',newList)
    groupSize=dict()
    dfData = pd.DataFrame(newList,columns=['Country','Year']).sort_values(by=['Year'],ascending=True)
    k=dfData.groupby('Year')
    for key,group in k:
        #print(key,group.size)
        groupSize[key] =(group.size)
    #print(list(groupSize.keys()))
    plt.figure(figsize=(16,6))
    sns.barplot(x=list(groupSize.keys()),y= list(groupSize.values()),palette='spring')
    #plt.xticks(list(groupSize.keys()),list(groupSize.keys()))
    plt.xlabel('Year',fontsize=18)
    plt.ylabel('Frequency',fontsize=18)
    plt.title('Total Participation of ' + each +   ' in atheletic events w.r.t. to Year',fontsize=18)
    plt.show()
df= pd.read_csv('../input/athlete_events.csv')
data= df.groupby('Sex')
attList=dict()
for name,count in data:
    attList[name]=count.size
plt.figure(figsize=(16,6))
sns.barplot(x=list(attList.keys()),y=list(attList.values()))
plt.xlabel('SEX',fontsize=18)
plt.ylabel('Total Contribution',fontsize=18)
plt.title('Total Participation',fontsize=18)
plt.show()
df= pd.read_csv('../input/athlete_events.csv').fillna(0).head(100)
df= df[df['Medal']!=0]
dfName = df['Name']
dfMedal= df['Medal']
#display(dfMedal)
dfCombined = pd.concat([dfName,dfMedal],axis=1)
#display(dfCombined)
dfCombinedNew= dfCombined.groupby('Name')
dictNew=dict()
for key,group in dfCombinedNew:
    dictNew[key]=(group.size)
dpNew= pd.DataFrame(list(dictNew.values()),index=[list(dictNew.keys())],columns =['No_Of_Medals_Won'])   
#display(dpNew)
plt.figure(figsize=(20,20))
sns.barplot(x=dpNew.index,y=dpNew['No_Of_Medals_Won'],palette='winter')
plt.show()
