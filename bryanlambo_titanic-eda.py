#Import the necessary libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
titanic=pd.read_csv('../input/titanicdataset-traincsv/train.csv')

titanic.head()
titanic.shape
titanic.info()
#Get the number of survivors per class

survived_per_class=titanic.groupby('Pclass').Survived.sum()

survived=survived_per_class.values

classes=survived_per_class.index



#Plot the number of survivors per class

plt.figure(1,figsize=(12,5))

plt.grid()

plt.suptitle('Which passenger class has the maximum number of survivors?')

plt.subplot(121)

plt.bar(classes,survived)

plt.xlabel('Passenger class')

plt.ylabel('Number of survivors')

plt.title('Number of survivors per class')

plt.xticks(classes)



#Plot the percentage of survivors per class

percent_survived_per_class=titanic.groupby('Pclass').Survived.sum()/titanic.Pclass.value_counts()

survived_percent=percent_survived_per_class.values

plt.subplot(122)

plt.bar(classes,survived_percent)

plt.xlabel('Passenger class')

plt.ylabel('Survivors percentage')

plt.title('Percentage of survivors per class')

plt.xticks(classes)
#Plot the number of survivors based on gender per class

male_surv = titanic[titanic.Sex=='male'].groupby('Pclass').Survived.sum()

female_surv = titanic[titanic.Sex=='female'].groupby('Pclass').Survived.sum()

plt.figure(2,figsize=(12,5))

plt.suptitle('What is the distribution, based on gender, of the survivors among the classes?')

plt.subplot(121)

width=0.3

plt.bar(classes, male_surv, width=width, label='males')

plt.bar(classes+width, female_surv, width=width, label='females')

plt.xticks(classes+width/2,classes) #This is to ensure that ticks is centered between 2 bars

plt.title('Male and Female survivors based on class')

plt.ylabel('Survivors')

plt.xlabel('Passenger class')

plt.legend()



#Plot the equivalent in percentage

plt.subplot(122)

width=0.3

total_male=titanic[titanic.Sex=='male'].groupby('Pclass').Pclass.count()

total_fem=titanic[titanic.Sex=='female'].groupby('Pclass').Pclass.count()

plt.bar(classes, male_surv/total_male, width=width, label='males')

plt.bar(classes+width, female_surv/total_fem, width=width, label='females')

plt.xticks(classes+width/2,classes) #This is to ensure that ticks is centered between 2 bars

plt.title('Male and Female survivors percentage based on class')

plt.ylabel('Survivors percentage')

plt.xlabel('Passenger class')

plt.legend()
#Create a plot of nonsurvivors with family per class

plt.figure(3,figsize=(12,5))

nonsurvivors_with_family=titanic[(titanic['SibSp']>0)|(titanic['Parch']>0) & (titanic['Survived']==0)].groupby('Pclass').Survived.count()

nonsurvivors=nonsurvivors_with_family.values

plt.subplot(121)

plt.bar(classes,nonsurvivors)

plt.title('Total number of nonsurvivors with family based on class')

plt.xlabel('Passenger class')

plt.ylabel('Number of nonsurvivors')

plt.xticks(classes)



#Plot the equivalent in percentage

percent_nonsurvivors=nonsurvivors/titanic.Pclass.value_counts().sort_index()

plt.subplot(122)

plt.bar(classes,percent_nonsurvivors)

plt.title('Percentage of nonsurvivors with family based on class')

plt.xlabel('Passenger class')

plt.ylabel('Percentage of nonsurvivors')

plt.xticks(classes)

plt.suptitle('What is the distribution of nonsurvivors among the various classes who have family aboard the ship?')
#Remember that the age group contains a lot of null values. So we first remove the null values.

titanic=titanic[np.isfinite(titanic['Age'])]



#Then we segregate the age accordingly using bins

age_bins=[0,18,25,40,60,100]

titanic['AgeBins']=pd.cut(titanic.Age,bins=age_bins)



#Plot total survivors per age category

ages=titanic.AgeBins.value_counts()

percent_ages=ages/titanic.AgeBins.count()

plt.figure(4,figsize=(12,5))

plt.subplot(121)

plt.pie(percent_ages.values, labels=percent_ages.index,autopct='%.1f%%')

plt.title('Total Passengers in different age groups')



#Survivors in each age group

survivors=titanic.groupby('AgeBins').Survived.sum()

plt.subplot(122)

#Pie automatically converts to percentages

plt.pie(survivors, labels=survivors.index,autopct='%.1f%%')

plt.title('Survivors in different age groups')