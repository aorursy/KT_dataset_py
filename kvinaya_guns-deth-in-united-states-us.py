# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
guns=pd.read_csv("../input/guns.csv")

guns.head()
guns=guns.dropna(axis=0,how='any')
guns.shape
gun=guns.loc[guns['sex']=='M']

gun['intent'].value_counts()
gun=guns.loc[guns['sex']=='F']

gun['intent'].value_counts()
count=guns.groupby(guns['intent'])

count['sex'].value_counts()

gun=guns.loc[guns['intent']=='Suicide']

gun['race'].value_counts()

gp=guns.groupby(guns['intent'])

gp['year'].value_counts()
# sex count in dataset

guns.sex.value_counts().plot(kind='bar')

plt.title('Sex count in dataset',fontsize=16,fontweight='bold')

plt.xlabel('sex',fontsize=14)

plt.ylabel('count',fontsize=14)

plt.xticks(rotation=0)

plt.show()
#race count

from matplotlib.pyplot import pie, axis, show

sums = guns.education.groupby(guns.race).count()

axis('equal');

pie(sums, labels=sums.index);

show()
intent_sex = guns.groupby(['intent', 'month'])['intent'].count().unstack('intent')

ax = intent_sex.plot(kind='bar', stacked=True, alpha=0.7)

ax.set_xlabel('intent', fontsize=14)

ax.set_ylabel('count', fontsize=14)

plt.xticks(rotation=0)

plt.title(' intent vs month distribution\nGun Deaths US: 2012-2014', fontsize=14, fontweight='bold')

sns.despine()

plt.show()

intent_sex = guns.groupby(['intent', 'year'])['intent'].count().unstack('intent')

ax = intent_sex.plot(kind='bar', stacked=True, alpha=0.7)

ax.set_xlabel('intent', fontsize=14)

ax.set_ylabel('count', fontsize=14)

plt.xticks(rotation=0)

plt.title(' intent vs year distribution\nGun Deaths US: 2012-2014', fontsize=14, fontweight='bold')

sns.despine()

plt.show()

g = sns.FacetGrid(guns, col="intent")  

g.map(sns.distplot, "age")  
guns.intent.value_counts().plot(kind='bar')

plt.title('intent count in dataset',fontsize=16)

plt.xlabel('intent',fontsize=16)

plt.ylabel('count',fontsize=16)

plt.xticks(rotation=0)

plt.show()



plt.hist(guns['age'])

plt.xlabel('age',fontsize=12)

plt.ylabel('count',fontsize=12)

plt.title('Age Distribution',fontsize=16,fontweight='bold')

plt.show()
suicide = guns[guns['intent'] == 'Suicide']

g=sns.FacetGrid(suicide, col='sex')  

g.map(sns.distplot, 'age')
suicide = guns[guns['intent'] == 'Suicide']

g=sns.FacetGrid(suicide, col='race',row='sex')  

g.map(sns.distplot,'age')
homicide = guns[guns['intent'] == 'Homicide']

g=sns.FacetGrid(homicide, col='race',row='sex')  

g.map(sns.distplot,'age')
#sex count

guns.sex.value_counts().plot(kind='pie',autopct='%1.1f%%')

plt.axis('equal')

plt.xticks(rotation=0)

plt.show()

Accident=guns[guns["intent"]=="Accidental"].race.value_counts().plot(kind="pie",autopct='%1.1f%%')

plt.axis('equal')

plt.title('Accident vs race',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)

plt.show()
#sucide vs age distribution

plt.hist(guns.age[guns.intent=='Suicide'],range(0,100))

plt.xlabel('age')

plt.ylabel('count')

plt.title('Age wise Suicide distribution\nGun Deaths US: 2012-2014')

plt.show()
plt.hist(guns.age[guns.intent=='Accidental'],range(0,100))

plt.xlabel('age')

plt.ylabel('count')

plt.title('Age wise Accidental distribution\nGun Deaths US: 2012-2014')

plt.show()
#homicide vs age distribution

plt.hist(guns.age[guns.intent=='Homicide'],range(0,100))

plt.xlabel('age')

plt.ylabel('count')

plt.title('Age wise Homicide distribution\nGun Deaths US: 2012-2014')

plt.show()
intent_sex = guns.groupby(['intent', 'sex'])['intent'].count().unstack('sex')

ax = intent_sex.plot(kind='bar', stacked=True, alpha=0.7)

ax.set_xlabel('intent', fontsize=14)

ax.set_ylabel('count', fontsize=14)

plt.xticks(rotation=0)

plt.title('Gender distribution\nGun Deaths US: 2012-2014', fontsize=14, fontweight='bold')

sns.despine()

plt.show()

intent_education = guns.groupby(['intent','education'])['intent'].count().unstack('education')

ax = intent_education.plot(kind='bar')

ax.set_xlabel('intent')

ax.set_ylabel('Count')

plt.xticks(rotation=0)

plt.title('Gender distribution\nGun Deaths US: 2012-2014', fontsize=14, fontweight='bold')

sns.despine()

plt.show()

#accident in race

fig, ax = plt.subplots(nrows=2,ncols=2)

fig.subplots_adjust(right=1.5,top=1,wspace = 0.2,hspace = 0.5 )

plt.subplot(2,2,1)

Accident=guns[guns["intent"]=="Accidental"].race.value_counts().plot(kind="pie")

plt.axis('equal')

plt.title('Accident vs race',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)





#suicide in race

plt.subplot(2,2,2)

suicide=guns[guns["intent"]=="Suicide"].race.value_counts().plot(kind="pie")

plt.axis('equal')

plt.title('Suicide vs race',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)





#homicide in race

plt.subplot(2,2,3)

Homicide=guns[guns["intent"]=="Homicide"].race.value_counts().plot(kind="pie")

plt.axis('equal')

plt.title('Homicide vs race',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)





#undermined in race

plt.subplot(2,2,4)

Undetermined=guns[guns["intent"]=='Undetermined'].race.value_counts().plot(kind="pie")

plt.axis('equal')

plt.title('Undetermined vs race',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)

plt.show()
fig, ax = plt.subplots(nrows=2,ncols=2)

plt.subplot(2,2,1)

labels="M","F"

colors=['lightblue','lightpink']

Accident=guns[guns["intent"]=="Accidental"].sex.value_counts()

plt.pie(Accident,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('Accident vs sex',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)





#suicide vs sex

plt.subplot(2,2,2)

suicide=guns[guns["intent"]=="Suicide"].sex.value_counts()

plt.pie(suicide,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('Suicide vs sex',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)





#homicide vs sex

plt.subplot(2,2,3)

Homicide=guns[guns["intent"]=="Homicide"].sex.value_counts()

plt.pie(Homicide,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('Homicide vs sex',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)





#undermined vs sex

plt.subplot(2,2,4)

sizes=[40,15]

Undetermined=guns[guns["intent"]=='Undetermined'].sex.value_counts()

plt.pie(Undetermined,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('Undetermined vs sex',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)

plt.show()

#female count for intent

labels='Suicide','Undetermined','Accidental','Homicide'

colors = ['gold','lightgreen','lightblue','orange']

fem_intent=guns[guns["sex"]=="M"].intent.value_counts()

plt.pie(fem_intent,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('female vs intent',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)

plt.show()



#male count for intent

male_intent=guns[guns["sex"]=="F"].intent.value_counts()

plt.pie(male_intent,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title('male vs intent',fontsize=15,fontweight="bold")

plt.xticks(rotation=0)

plt.show()

fig, ax = plt.subplots(nrows=1,ncols=3)

fig.subplots_adjust(right=1.5,top=1,wspace = 0.4,hspace = 0.5 )

plt.subplot(1,3,1)

labels='Suicide','Undetermined','Accidental','Homicide'

colors=['gold','lightgreen','lightpink','silver']

year=guns[guns["year"]==2012].intent.value_counts()

plt.pie(year,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('intent count in 2012',fontsize=16,fontweight='bold')

plt.axis('equal')

plt.xticks(rotation=0)



plt.subplot(1,3,2)

year=guns[guns["year"]==2013].intent.value_counts()

plt.pie(year,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('intent count in 2013',fontsize=16,fontweight='bold')

plt.axis('equal')

plt.xticks(rotation=0)





plt.subplot(1,3,3)

year=guns[guns["year"]==2014].intent.value_counts()

plt.pie(year,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('intent count in 2014',fontsize=16,fontweight='bold')

plt.axis('equal')

plt.xticks(rotation=0)

plt.show()
