

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('../input/StudentsPerformance.csv')

# Any results you write to the current directory are saved as output.
data.head()
data.info()
data.corr()
data.describe()
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)
plt.show()


sns.jointplot(x=data['reading score'],y=data['writing score'],kind='reg')
plt.xlabel('reading score')
plt.ylabel('writing score')
plt.title('READİNG WRİTİNG SCORES')
plt.show()
data.boxplot('math score',by='gender')

plt.title('math&gender')
print(data.loc[:,['gender','writing score']].head())
plt.hist(data['math score'])
plt.xlabel('puan')
plt.ylabel('frekans')
plt.title('MATEMATİK PUANI FREKANSLARI')
plt.show
plt.hist(data['writing score'])
plt.xlabel('puan')
plt.ylabel('frekans')
plt.title('WRİTİNG PUANI FREKANSLARI')
plt.show
plt.hist(data['reading score'])
plt.xlabel('puan')
plt.ylabel('frekans')
plt.title('READİNG PUANI FREKANSLARI')
plt.show
data['total']=data['math score']+data['writing score']+data['reading score']
data.head()
data2=data.pivot_table(index='gender',columns='race/ethnicity',values='math score')
data2

data2=data.pivot_table(index='gender',columns='lunch',values='math score')
data2#lunch type
data2=data.groupby('gender').mean()
data2
#female>male :)

math=[]
read=[]
write=[]
cins=list(data['gender'].unique())
for i in cins:
    x=data['gender']==i

    y=data['math score'][x]
    z = sum(y) / len(y)
    math.append(z)

    y1=data['reading score'][x]
    z1=sum(y1)/len(y1)
    read.append(z1)

    y2=data['writing score'][x]
    z2=sum(y2)/len(y2)
    write.append(z2)

frame=pd.DataFrame({'gender':cins,'math':math,'read':read,'write':write})


liste=['math','read','write']
frame.set_index('gender',inplace=True)


sns.pointplot(x=liste,y=frame.iloc[0,:],color='r')
sns.pointplot(x=liste,y=frame.iloc[1,:],color='b')

plt.xlabel('male',fontsize=12,color='b')
plt.ylabel('SCORES',color='g')
plt.title('female',color='r')


plt.ylabel('scores')


plt.grid()
plt.show()

data3=data.groupby('race/ethnicity').mean()
data3
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.barplot(x = 'gender', y = 'reading score', data = data)
plt.title('READİNG',color='red')

plt.subplot(1,3,2)
sns.barplot(x = 'gender', y = 'writing score', data = data)
plt.title('WRİTİNG',color='red')


plt.subplot(1,3,3)
sns.barplot(x = 'gender', y = 'math score', data = data)
plt.title('MATH',color='red')

plt.show()

plt.figure(figsize=(15,8))

plt.subplot(1,3,1)
sns.barplot(y = 'race/ethnicity', x = 'reading score', data = data,hue='gender')
plt.title('READİNG',color='red')

plt.subplot(1,3,2)
sns.barplot(y = 'race/ethnicity', x = 'writing score', data = data,hue='gender')
plt.title('WRİTİNG',color='red')


plt.subplot(1,3,3)
sns.barplot(y = 'race/ethnicity', x = 'math score', data = data,hue='gender')
plt.title('MATH',color='red')

plt.show()


sayi=data['gender'].value_counts().values
labels=data['gender'].value_counts().index
exp=[0,0]
color=['orange','lime']
plt.pie(sayi,explode=exp,labels=labels,autopct='%1.1f')
plt.title('GENDER TABLE',color='purple')
plt.show()
sayi=data['race/ethnicity'].value_counts().values
labels=data['race/ethnicity'].value_counts().index
exp=[0,0,0,0,0]

plt.pie(sayi,explode=exp,labels=labels,autopct='%1.1f')
plt.title('GROUP TABLE',color='purple')
plt.show()

sayi=data['parental level of education'].value_counts().values
labels=data['parental level of education'].value_counts().index
exp=[0,0,0,0,0,0]

plt.pie(sayi,explode=exp,labels=labels,autopct='%1.1f')
plt.title('EDUCATİON TABLE',color='purple')
plt.show()
sayi=data['lunch'].value_counts().values
labels=data['lunch'].value_counts().index
exp=[0,0]

plt.pie(sayi,explode=exp,labels=labels,autopct='%1.1f')
plt.title('LUNCH TABLE',color='purple')

plt.show()

plt.figure(figsize=(8,8))
sns.boxplot(x=data['gender'],y=data['math score'],hue=data['lunch'])
plt.title('LUNCH & MATH SCORE')

plt.show()

plt.figure(figsize=(5,8))
sns.swarmplot(x=data['gender'],y=data['math score'],hue=data['test preparation course'])

plt.title('TEST & MATH SCORE')

plt.show()

sns.countplot(data['race/ethnicity'],hue=data['gender'])
plt.title('gender&counts&group')
plt.show()