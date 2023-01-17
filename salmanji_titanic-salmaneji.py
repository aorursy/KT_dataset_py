# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.size
df.head()
df.tail()
df.dtypes
df.columns
df.describe()
df.isnull().sum().sort_values(ascending = False)
df.isnull().sum().sum()
import matplotlib.pyplot as plt
plt.style.use('classic')
fig=plt.figure()
axes=plt.axes()
plt.title("Representation of features: Survived")
marqueur=[0.25,0.75]
y=['Not Survived','Survived']
plt.xticks(marqueur,y)
plt.hist(df['Survived'],bins=2)
nb_Not_Surv=df[df['Survived']==0].size
nb_Surv=df[df['Survived']==1].size
fig, axes = plt.subplots(1, 2, figsize=(10,3))
axes[0].set_title("Histogram of features: Survived")
x_pos=[0.25,0.75]
x=['Not Survived','Survived']
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(x)
axes[0].hist(df['Survived'],bins=2)
labels='Not Survived','Survived'
size=[nb_Not_Surv,nb_Surv]
explode = (0, 0.1)
axes[1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',)
axes[1].set_title("Pie of features: Survived")
import seaborn as sns
fig=plt.figure()
axes=plt.axes()
graphe=sns.countplot(x='Survived',data=df,hue='Sex').set_title('Representation of Number of Survived By Sex')
axes.set_xticklabels(['Death','Survived'])
df[(df['Sex']=='female')&(df['Age']>25)]
df[(df['Sex']=='female')&(df['Age']>25)].size
df[(df['Pclass']==1)&(df['Survived']==1)].size
plt.style.use('classic')
fig=plt.figure()
axes=plt.axes()
plt.title("Representation of features: Age")
plt.hist(df['Age'],bins=5,color='skyblue')
graphes=df.hist(column='Age',by='Sex',bins=5,layout = (1, 2))
plt.suptitle('Representation the age of the passenger By Sex', x=0.5, y=1.05, ha='center', fontsize='xx-large')
for ax in graphes.flatten():
    ax.set_xlabel("Age")
    ax.set_ylabel("Individuals")
sns.violinplot(x="Sex", y="Age", data=df,palette='rainbow').set_title('Representation the age of the passenger By Sex')
name=df['Name']
name[0]
pos1 = name[0].find('Mr')
name[0][pos1:pos1+2]
title=['Mr','Miss','Major','Mlle','Mme']
tab_title=[]
for nom in name:
    posMr=nom.find('Mr')
    posMiss=nom.find('Miss')
    posMlle=nom.find('Mlle')
    posMajor=nom.find('Major')
    posMme=nom.find('Mme')
    if posMr!=-1:
        tab_title.append(nom[posMr:posMr+2])
    elif posMiss!=-1:
        tab_title.append(nom[posMiss:posMiss+4])
    elif posMlle!=-1:
        tab_title.append(nom[posMlle:posMlle+4])
    elif  posMajor!=-1:
        tab_title.append(nom[posMajor:posMajor+5])
    elif  posMme!=-1:
        tab_title.append(nom[posMme:posMme+3])
tab_title
fig=plt.figure()
axes=plt.axes()
#1er Graphe
plt.title("Number of Passengers by Pclass")
x_pos=[1.25,2,2.75]
x=['First Class','Second Class','Third Class']
plt.xticks(x_pos,x)
plt.hist(df['Pclass'],bins=3)

#2ieme graphe
graphes=df.hist(column='Pclass',by='Survived',bins=3)
plt.suptitle('Survived VS Not Survived by class', x=0.5, y=1.05, ha='center', fontsize='xx-large')
for ax in graphes.flatten():
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_xlabel("PClass")
#3ieme graphe
#
dfSurv=df[df['Survived']==1]
dfSurv.hist(column='Pclass',by='Survived',bins=3)
plt.title("Number of Survived by class")
x_pos=[1.25,2,2.75]
x=['First Class','Second Class','Third Class']
plt.xticks(x_pos,x)
dfDead=df[df['Survived']==0]
fig, axes = plt.subplots(1, 3, figsize=(25,10))
x_pos=[1.25,2,2.75]
x=['First Class','Second Class','Third Class']
axes[0].set_title("Number of Passengers by Pclass")
axes[1].set_title("Dead passenger by Pclass")
axes[2].set_title("Survived by Pclass")
axes[0].set_ylabel("Number of passengers")
axes[1].set_ylabel("Number of DEAD passengers")
axes[2].set_ylabel("Number of SURVIVED passengers")
for i in range(3):
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(x)
    axes[i].set_xlabel("PClass")
axes[0].hist(df['Pclass'],bins=3)  
axes[1].hist(dfDead['Pclass'],bins=3)
axes[2].hist(dfSurv['Pclass'],bins=3)
#!conda install -c conda-forge pandas-profiling
import pandas_profiling as pdp
pdp.ProfileReport(df)
df.corr()
sns.heatmap(df.corr(),cmap='coolwarm')
plt.title('Correlation between features of Data : TITANIC ')
print('The TOTAL NUMBER of passenger of titanic =',df.size)
print('_'*50)
print('    * Number of Women in titanic =',df[df['Sex']=='female'].size)
print('    * Number of Men in titanic =',df[df['Sex']=='male'].size)
print('_'*50)
print('    * Number of Survival =',df[df['Survived']==1].size)
print('    * Number of Death =',df[df['Survived']==0].size)
print('_'*50)
print('    * Number of passenger on 1st Class =',df[df['Pclass']==1].size)
print('    * Number of passenger on 2d Class =',df[df['Pclass']==2].size)
print('    * Number of passenger on 3d Class =',df[df['Pclass']==3].size)
print('_'*50)
nb_enfant=df[df['Age']<17].size
nb_jeune=df[(df['Age']>=17)&(df['Age']<26)].size
nb_adulte=df[(df['Age']>=26)&(df['Age']<55)].size
nb_vieux=df[df['Age']>=55].size
print('    * Number of KIDS passenger < 17 =',nb_enfant)
print('    * Number of YOUNG ADULT passenger between 17 and 25  =',nb_jeune)
print('    * Number of ADULT passenger between 26 and 54  =',nb_adulte)
print('    * Number of KIDS passenger >54 =',nb_vieux)
print('MISSING DATA : ')
MissData=df.isnull().sum()
print(' * TOTAL of missing data= ',MissData.sum())
for i in MissData.index:
    nb=100*(MissData[i]/MissData.sum())
    print('       -For {0:15} Percentage of Missing value = {1:.2f} %'.format(i,nb))
fig, axes = plt.subplots(2, 2, figsize=(20,10))
fig.suptitle(' * Resume of features :', fontsize=40)
nb_Not_Surv=df[df['Survived']==0].size
nb_Surv=df[df['Survived']==1].size
labels='Not Survived','Survived'
size=[nb_Not_Surv,nb_Surv]
explode = (0, 0.1)
axes[0][0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
axes[0][0].set_title('Percentage Death & Survived')
nb_male=df[df['Sex']=='male'].size
nb_female=df[df['Sex']=='female'].size
labels='male','female'
size=[nb_male,nb_female]
explode = (0, 0.1)
axes[0][1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
axes[0][1].set_title('Percentage Male & Female')
nb_c1=df[df['Pclass']==1].size
nb_c2=df[df['Pclass']==2].size
nb_c3=df[df['Pclass']==3].size
labels='1st Class','2d Class','3d Class'
size=[nb_c1,nb_c2,nb_c3]
explode = (0, 0.1,0)
axes[1][0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
axes[1][0].set_title('Percentage Class')
nb_enfant=df[df['Age']<17].size
nb_jeune=df[(df['Age']>=17)&(df['Age']<26)].size
nb_adulte=df[(df['Age']>=26)&(df['Age']<55)].size
nb_vieux=df[df['Age']>=55].size
labels='Young','Young Adult','Adult','Old'
size=[nb_enfant,nb_jeune,nb_adulte,nb_vieux]
explode = (0, 0.1,0,0)
axes[1][1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
axes[1][1].set_title('Passenger by Age')
fig, axes = plt.subplots(1, 2, figsize=(20,10))
fig.suptitle(' * Number of survived or dead by sex :', fontsize=40)
nb_MaleSur=df[(df['Survived']==1)&(df['Sex']=='male')].size
nb_FemSur=df[(df['Survived']==1)&(df['Sex']=='female')].size
labels='Male','Female'
size=[nb_MaleSur,nb_FemSur]
explode = (0, 0.1)
axes[0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[0].set_title('Survived by sex')

nb_MaleD=df[(df['Survived']==0)&(df['Sex']=='male')].size
nb_FemD=df[(df['Survived']==0)&(df['Sex']=='female')].size
labels='Male','Female'
size=[nb_MaleD,nb_FemD]
explode = (0, 0.1)
axes[1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[1].set_title('Death by sex')
fig, axes = plt.subplots(1, 2, figsize=(20,10))
fig.suptitle(' * Number of survived or dead by class :', fontsize=40)
nb_c1=df[(df['Survived']==1)&(df['Pclass']==1)].size
nb_c2=df[(df['Survived']==1)&(df['Pclass']==2)].size
nb_c3=df[(df['Survived']==1)&(df['Pclass']==3)].size
labels='1st Class','2d Class','3d Class'
size=[nb_c1,nb_c2,nb_c3]
explode = (0.1, 0,0)
axes[0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[0].set_title('Survived by class')

nb_c1=df[(df['Survived']==0)&(df['Pclass']==1)].size
nb_c2=df[(df['Survived']==0)&(df['Pclass']==2)].size
nb_c3=df[(df['Survived']==0)&(df['Pclass']==3)].size
labels='1st Class','2d Class','3d Class'
size=[nb_c1,nb_c2,nb_c3]
explode = (0.1, 0,0)
axes[1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[1].set_title('Death by class')
fig, axes = plt.subplots(1, 2, figsize=(20,10))
fig.suptitle(' * Number of survived or dead by age :', fontsize=40)
nb_enfant=df[(df['Survived']==1)&(df['Age']<17)].size
nb_jeune=df[(df['Survived']==1)&(df['Age']>=17)&(df['Age']<26)].size
nb_adulte=df[(df['Survived']==1)&(df['Age']>=26)&(df['Age']<55)].size
nb_vieux=df[(df['Survived']==1)&(df['Age']>=55)].size
labels='Young','Young Adult','Adult','Old'
size=[nb_enfant,nb_jeune,nb_adulte,nb_vieux]
explode = (0, 0.1,0,0)
axes[0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[0].set_title('Passenger Survived by Age')

nb_enfant=df[(df['Survived']==0)&(df['Age']<17)].size
nb_jeune=df[(df['Survived']==0)&(df['Age']>=17)&(df['Age']<26)].size
nb_adulte=df[(df['Survived']==0)&(df['Age']>=26)&(df['Age']<55)].size
nb_vieux=df[(df['Survived']==0)&(df['Age']>=55)].size
labels='Young','Young Adult','Adult','Old'
size=[nb_enfant,nb_jeune,nb_adulte,nb_vieux]
explode = (0, 0.1,0,0)
axes[1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[1].set_title('Passenger Dead by Age')
nb_c1=df[df['Embarked']=='C'].size
nb_c2=df[df['Embarked']=='S'].size
nb_c3=df[df['Embarked']=='Q'].size
labels='Cherbourg','Southampton','Queenstown'
size=[nb_c1,nb_c2,nb_c3]
explode = (0, 0.1,0)
plt.pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',)
plt.title('Percentage of Embarked')
fig, axes = plt.subplots(1, 2, figsize=(20,10))
fig.suptitle(' * Number of survived or dead by embarked :', fontsize=40)
nb_c1=df[(df['Survived']==1)&(df['Embarked']=='C')].size
nb_c2=df[(df['Survived']==1)&(df['Embarked']=='S')].size
nb_c3=df[(df['Survived']==1)&(df['Embarked']=='Q')].size
labels='Cherbourg','Southampton','Queenstown'
size=[nb_c1,nb_c2,nb_c3]
explode = (0, 0.1,0)
axes[0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[0].set_title('Percentage Survived Embarked')
nb_c1=df[(df['Survived']==0)&(df['Embarked']=='C')].size
nb_c2=df[(df['Survived']==0)&(df['Embarked']=='S')].size
nb_c3=df[(df['Survived']==0)&(df['Embarked']=='Q')].size
labels='Cherbourg','Southampton','Queenstown'
size=[nb_c1,nb_c2,nb_c3]
explode = (0, 0.1,0)
axes[1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[1].set_title('Percentage Death Embarked')
fig=plt.figure()
axes=plt.axes()
plt.title(' * Number of passenger by SibSp :', fontsize=40)
sns.countplot(x='SibSp',data=df, hue='SibSp')
plt.legend(loc='upper right')
fig=plt.figure()
axes=plt.axes()
plt.title(' * Number of survived or dead by SibSp :', fontsize=40)
sns.countplot(x='Survived',data=df, hue='SibSp')
plt.legend(loc='upper right')
axes.set_xticklabels(['Death','Survived'])
fig=plt.figure()
axes=plt.axes()
plt.title(' * Number of survived or dead by Parch :', fontsize=40)
sns.countplot(x='Survived',data=df, hue='Parch')
plt.legend(loc='upper right')
axes.set_xticklabels(['Death','Survived'])
fig=plt.figure()
axes=plt.axes()
plt.title("Representation of features: Fare")
plt.hist(df['Fare'],bins=50)
nb_f=((df[df['Fare']<35].size)/df.size)*100
print('The percentage of Fare <35  = {0:.2f} %'.format(nb_f))
nb_Not_Surv=df[df['Survived']==0].size
nb_Surv=df[df['Survived']==1].size
labels='Not Survived','Survived'
size=[nb_Not_Surv,nb_Surv]
explode = (0, 0.1)
plt.pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
plt.title('Percentage Death & Survived')
nb_Not_Surv=df[df['Survived']==0].size
nb_Surv=df[df['Survived']==1].size
labels='Not Survived','Survived'
size=[nb_Not_Surv,nb_Surv]
explode = (0, 0.1)
plt.pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
plt.title('Percentage Death & Survived')
fig, axes = plt.subplots(2, 3, figsize=(20,20))
fig.suptitle(' * Resume of features SEX AND Pclass :', fontsize=40)
nb_male=df[df['Sex']=='male'].size
nb_female=df[df['Sex']=='female'].size
labels='male','female'
size=[nb_male,nb_female]
explode = (0, 0.1)
axes[0][0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
axes[0][0].set_title('Percentage Male & Female')

nb_MaleSur=df[(df['Survived']==1)&(df['Sex']=='male')].size
nb_FemSur=df[(df['Survived']==1)&(df['Sex']=='female')].size
labels='Male','Female'
size=[nb_MaleSur,nb_FemSur]
explode = (0, 0.1)
axes[0][1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[0][1].set_title('Survived by sex')

nb_MaleD=df[(df['Survived']==0)&(df['Sex']=='male')].size
nb_FemD=df[(df['Survived']==0)&(df['Sex']=='female')].size
labels='Male','Female'
size=[nb_MaleD,nb_FemD]
explode = (0, 0.1)
axes[0][2].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[0][2].set_title('Death by sex')

nb_c1=df[df['Pclass']==1].size
nb_c2=df[df['Pclass']==2].size
nb_c3=df[df['Pclass']==3].size
labels='1st Class','2d Class','3d Class'
size=[nb_c1,nb_c2,nb_c3]
explode = (0, 0.1,0)
axes[1][0].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'x-large'})
axes[1][0].set_title('Percentage Class')

nb_c1=df[(df['Survived']==1)&(df['Pclass']==1)].size
nb_c2=df[(df['Survived']==1)&(df['Pclass']==2)].size
nb_c3=df[(df['Survived']==1)&(df['Pclass']==3)].size
labels='1st Class','2d Class','3d Class'
size=[nb_c1,nb_c2,nb_c3]
explode = (0.1, 0,0)
axes[1][1].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[1][1].set_title('Survived by class')

nb_c1=df[(df['Survived']==0)&(df['Pclass']==1)].size
nb_c2=df[(df['Survived']==0)&(df['Pclass']==2)].size
nb_c3=df[(df['Survived']==0)&(df['Pclass']==3)].size
labels='1st Class','2d Class','3d Class'
size=[nb_c1,nb_c2,nb_c3]
explode = (0.1, 0,0)
axes[1][2].pie(size,labels=labels,shadow=True,startangle=90,explode=explode,autopct='%1.1f%%',textprops={'size': 'xx-large'})
axes[1][2].set_title('Death by class')
nb_fem=((df[(df['Survived']==1)&(df['Pclass']==1)&(df['Sex']=='female')].size)/df.size)*100
print('For Women in 1st class the chance of surviving to Titanic  = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==1)&(df['Pclass']==1)&(df['Sex']=='male')].size)/df.size)*100
print('For Men in 1st class the chance of surviving to Titanic  = {0:.2f} %'.format(nb_hom))

nb_fem=((df[(df['Survived']==0)&(df['Pclass']==1)&(df['Sex']=='female')].size)/df.size)*100
print('For Women in 1st class the chance of dead to Titanic  = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==0)&(df['Pclass']==1)&(df['Sex']=='male')].size)/df.size)*100
print('For Men in 1st class the chance of dead to Titanic  = {0:.2f} %'.format(nb_hom))
nb_hom=((df[(df['Survived']==1)&(df['Pclass']==2)&(df['Sex']=='male')].size)/df.size)*100
print('For Men in 2d class the chance of surviving to Titanic  = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==1)&(df['Pclass']==2)&(df['Sex']=='female')].size)/df.size)*100
print('For Women in 2d class the chance of surviving to Titanic  = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==0)&(df['Pclass']==2)&(df['Sex']=='male')].size)/df.size)*100
print('For Men in 2d class the chance of dead to Titanic  = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==0)&(df['Pclass']==2)&(df['Sex']=='female')].size)/df.size)*100
print('For Women in 2d class the chance of dead to Titanic  = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==1)&(df['Pclass']==3)&(df['Sex']=='male')].size)/df.size)*100
print('For Men in 3d class the chance of surviving to Titanic  = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==1)&(df['Pclass']==3)&(df['Sex']=='female')].size)/df.size)*100
print('For Women in 3d class the chance of surviving to Titanic  = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==0)&(df['Pclass']==3)&(df['Sex']=='male')].size)/df.size)*100
print('For Men in 3d class the chance of dead to Titanic  = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==0)&(df['Pclass']==3)&(df['Sex']=='female')].size)/df.size)*100
print('For Women in 3d class the chance of dead to Titanic  = {0:.2f} %'.format(nb_fem))
nb_fem=((df[(df['Survived']==1)&(df['Pclass']==1)&(df['Sex']=='female')].size)/df[df['Pclass']==1].size)*100
print('Comparing to the number of passenger in 1st class, The percentage of survived Women = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==1)&(df['Pclass']==1)&(df['Sex']=='male')].size)/df[df['Pclass']==1].size)*100
print('Comparing to the number of passenger in 1st class, The percentage of survived Men = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==0)&(df['Pclass']==1)&(df['Sex']=='female')].size)/df[df['Pclass']==1].size)*100
print('Comparing to the number of passenger in 1st class, The percentage of dead Women = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==0)&(df['Pclass']==1)&(df['Sex']=='male')].size)/df[df['Pclass']==1].size)*100
print('Comparing to the number of passenger in 1st class, The percentage of dead Men = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==1)&(df['Pclass']==3)&(df['Sex']=='female')].size)/df[df['Pclass']==3].size)*100
print('Comparing to the number of passenger in 3d class, The percentage of survived women = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==1)&(df['Pclass']==3)&(df['Sex']=='male')].size)/df[df['Pclass']==3].size)*100
print('Comparing to the number of passenger in 3d class, The percentage of survived Men = {0:.2f} %'.format(nb_hom))
nb_fem=((df[(df['Survived']==0)&(df['Pclass']==3)&(df['Sex']=='female')].size)/df[df['Pclass']==3].size)*100
print('Comparing to the number of passenger in 3d class, The percentage of dead Women = {0:.2f} %'.format(nb_fem))
nb_hom=((df[(df['Survived']==0)&(df['Pclass']==3)&(df['Sex']=='male')].size)/df[df['Pclass']==3].size)*100
print('Comparing to the number of passenger in 3d class, The percentage of dead Men = {0:.2f} %'.format(nb_hom))
