# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization tools
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/WorldCupMatches.csv')
#filtering
finaldata = data[(data['Stage']=='Final')&(data['Year']>1950)]
finaldata = finaldata.reset_index(drop=True)
finaldata
for w in range(0,len(finaldata)):
    for k in range(w+1,len(finaldata)):
        if finaldata['Year'][w] == finaldata['Year'][k]:
            finaldata.drop(finaldata.index[w],inplace = True)
# we did delete same year
finaldata.head()
finaldata.tail()
ma = finaldata['Attendance'].idxmax(axis = 0,skipna = True) #most attendance gives index
# Most attendance
finaldata.iloc[ma]
finaldata.plot(kind = 'scatter',x='Year',y='Home Team Goals',alpha=.5,color='black')
plt.xlabel('Year')
plt.ylabel('Home Team Goals')
plt.title('Scatter')
plt.show()

finaldata.plot(kind = 'scatter',x='Year',y='Away Team Goals',alpha=.5,color='black')
plt.xlabel('Year')
plt.ylabel('Away Team Goals')
plt.title('Scatter')
plt.show()
data['Away Team Goals'].plot(kind='line',color='b',label='Home Team Goals',linewidth=2,alpha=.5,grid=True,linestyle='-')
data['Home Team Goals'].plot(color='r',label='Away Team Goals',linewidth=2,alpha=.5,grid=True,linestyle='-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
finaldata.corr()
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(finaldata.corr(),annot = True,linewidth = 3,fmt='.2f',ax=ax)
dataframe = pd.DataFrame(data,columns = ['Year','Stage','Stadium','Home Team Name','Home Team Goals','Away Team Goals','Away Team Name','Attendance'])
x = dataframe['Year'] == 2010
dataframe = dataframe[x] # we choose only 2010 world cup
dataframe = dataframe.reset_index(drop = True) # we reset the index of new dataframe
dataframe.info() # we see column name, data types, row number etc.
dataframe.describe() # some information with data set(max goal-min goal-max attendance etc.)
dataframe.corr() # Correlation of dataset
print(dataframe['Home Team Goals'].corr(dataframe['Attendance'])) # see home team goals correlation with attendance
print(dataframe['Away Team Goals'].corr(dataframe['Attendance'])) # see away team goals correlation with attendance
sns.heatmap(dataframe.corr(),annot = True,linewidths = 3,fmt = '.2f') # visualization correlation
ag = dataframe['Stage'] == 'Group A' # we created true-false statement
agroup = dataframe[ag] # new dataframe, only group A matches
agroup = agroup.reset_index(drop = True) # we reset the index of new dataframe
agroup = agroup.assign(totally_goal=(agroup['Home Team Goals']+agroup['Away Team Goals'])) # we created new columns. We will see total goals for every match 
agroup.describe() # information with A group
print("--GROUP A most goal match --")
for w in range(len(agroup)):
    if agroup['totally_goal'][w] == agroup['totally_goal'].max():
        print(agroup['Home Team Name'][w],":",agroup['Home Team Goals'][w])
        print(agroup['Away Team Name'][w],":",agroup['Away Team Goals'][w])
        print("----------------------------------")
    else:
        continue
agroup.totally_goal.plot(kind='hist',bins=20,figsize=(8,8))
# VİSUALİZATİON
f,ax = plt.subplots(figsize =(10,10)) # visualization of group A matches
sns.heatmap(agroup.corr(),annot = True,linewidths = 3,fmt = '.4f',ax=ax)
# Group B
bg = dataframe['Stage'] == 'Group B'
bgroup = dataframe[bg]
bgroup = bgroup.reset_index(drop = True)
bgroup = bgroup.assign(totally_goal=(bgroup['Home Team Goals']+bgroup['Away Team Goals']))
bgroup.describe()

print("--GROUP B most goal matches--")
for w in range(len(bgroup)):
    if bgroup['totally_goal'][w] == bgroup['totally_goal'].max():
        print(bgroup['Home Team Name'][w],":",bgroup['Home Team Goals'][w])
        print(bgroup['Away Team Name'][w],":",bgroup['Away Team Goals'][w])
        print("---------------------------------")
    else:
        continue
ros = dataframe['Stage'] == 'Round of 16'
roundof16 = dataframe[ros]
roundof16 = roundof16.reset_index(drop = True)
roundof16 = roundof16.assign(totally_goal=(roundof16['Home Team Goals']+roundof16['Away Team Goals']))
roundof16.describe()
print("--ROUND OF 16 most goal matches--")
for w in range(len(roundof16)):
    if roundof16['totally_goal'][w] == roundof16['totally_goal'].max():
        print(roundof16['Home Team Name'][w],":",roundof16['Home Team Goals'][w])
        print(roundof16['Away Team Name'][w],":",roundof16['Away Team Goals'][w])
        print("---------------------------------")
    else:
        continue
f,ax = plt.subplots(figsize =(10,10)) # son 16 turu görselleştirme
sns.heatmap(roundof16.corr(),annot = True,linewidths = 3,fmt = '.2f',ax=ax)
roundof16.totally_goal.plot(kind='hist',bins=20,figsize=(8,8))
qf = dataframe['Stage'] == 'Quarter-finals'
quaf = dataframe[qf]
quaf = quaf.reset_index(drop = True)
quaf = quaf.assign(totally_goal=(quaf['Home Team Goals']+quaf['Away Team Goals']))
quaf.describe()
print("--QUARTER FINAL most goal matches--")
for w in range(len(quaf)):
    if quaf['totally_goal'][w] == quaf['totally_goal'].max():
        print(quaf['Home Team Name'][w],":",quaf['Home Team Goals'][w])
        print(quaf['Away Team Name'][w],":",quaf['Away Team Goals'][w])
        print("---------------------------------")
    else:
        continue
f,ax = plt.subplots(figsize =(10,10)) #quarter final visualization
sns.heatmap(quaf.corr(),annot = True,linewidths = 3,fmt = '.2f',ax=ax)
sf = dataframe['Stage'] == 'Semi-finals'
semif = dataframe[sf]
semif = semif.reset_index(drop = True)
semif = semif.assign(totally_goal=(semif['Home Team Goals']+semif['Away Team Goals']))
print("--SEMI FINAL most goal matches--")
for w in range(len(semif)):
    if semif['totally_goal'][w] == semif['totally_goal'].max():
        print(semif['Home Team Name'][w],":",semif['Home Team Goals'][w])
        print(semif['Away Team Name'][w],":",semif['Away Team Goals'][w])
        print("---------------------------------")
    else:
        continue