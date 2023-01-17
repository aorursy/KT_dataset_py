import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/StudentsPerformance.csv')

data.head()
mathScore = []

readingScore = []

writingScore = []

groupList = list(data['race/ethnicity'].unique())



for i in groupList:

    tmp = data[data['race/ethnicity'] == i]

    averages = sum(tmp['math score']) / len(tmp)

    averages2 = sum(tmp['reading score']) / len(tmp)

    averages3 = sum(tmp['writing score']) / len(tmp)

    mathScore.append(averages)

    readingScore.append(averages2)

    writingScore.append(averages3)



#creating Data Frames

df1 = pd.DataFrame({'group': groupList, 'mathScore': mathScore})

df2 = pd.DataFrame({'group': groupList, 'readingScore': readingScore})

df3 = pd.DataFrame({'group': groupList, 'writingScore': writingScore})



#shorting

newindex = (df1['mathScore'].sort_values(ascending = True)).index.values

newindex2 = (df2['readingScore'].sort_values(ascending = True)).index.values

newindex3 = (df3['writingScore'].sort_values(ascending = True)).index.values

sorted_Data = df1.reindex(newindex)

sorted_Data2 = df2.reindex(newindex2)

sorted_Data3 = df3.reindex(newindex3)



#normalization

sorted_Data['mathScore'] = sorted_Data['mathScore'] / max(sorted_Data['mathScore'])

sorted_Data2['readingScore'] = sorted_Data2['readingScore'] / max(sorted_Data2['readingScore'])

sorted_Data3['writingScore'] = sorted_Data3['writingScore'] / max(sorted_Data3['writingScore'])

last_data = pd.concat([sorted_Data,sorted_Data2['readingScore'],sorted_Data3['writingScore']], axis = 1)

last_data.sort_values('mathScore', inplace = True)

last_data.sort_values('readingScore', inplace = True) 

last_data.sort_values('writingScore', inplace = True) 



# visualization

vs = plt.subplots(figsize = (15, 9))

sns.pointplot(x = 'group', y = 'mathScore', data = last_data, color = 'red', alpha = 0.8)

sns.pointplot(x = 'group', y = 'readingScore', data = last_data, color = 'blue', alpha = 0.8)

sns.pointplot(x = 'group', y = 'writingScore', data = last_data, color = 'green', alpha = 0.8)

plt.text(3.5, 0.89,'Math Scores',color='red',fontsize = 17,style = 'italic')

plt.text(3.5, 0.88,'Reading Scores',color='blue',fontsize = 17,style = 'italic')

plt.text(3.5, 0.87,'Writing Scores',color='green',fontsize = 17,style = 'italic')

plt.xlabel('Groups',fontsize = 15,color='black')

plt.ylabel('Values',fontsize = 15,color='black')

plt.title('Math Scores - Reading Scores - Writing Scores',fontsize = 20,color='black')

plt.grid()

last_data.head()
jp = sns.jointplot(last_data.mathScore, last_data.readingScore, kind='kde', height=8)

plt.savefig('graph.png')

plt.show()
jp2 = sns.jointplot('mathScore', 'readingScore', data=last_data, color='b', size=7, ratio=5)
#Lunch rates

data.lunch.dropna(inplace = True)

labels = data.lunch.value_counts().index

ratios = data.lunch.value_counts().values

colors = ['aqua','grey']

explode = [0,0]



plt.figure(figsize = (9,9))

plt.pie(ratios, labels = labels, colors = colors, autopct = '%1.0f%%')

plt.title('Lunch rates', color='blue',fontsize = 20)

plt.show()
sns.lmplot(x = "mathScore", y = "readingScore", data = last_data)

plt.show()