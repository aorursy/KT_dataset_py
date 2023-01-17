import numpy as np
import pandas as pd 
import random as rnd
from os import *
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
%matplotlib inline
import os

train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")


print(train_df.columns)
print(test_df.columns)


myvars = ['Sex', 'Survived', 'Pclass']
print(train_df[myvars].groupby(['Pclass']).mean())
print(train_df[['Sex', 'Survived']].groupby(['Sex']).mean())
CHILD = train_df['Age'] < 15
CREW = train_df['Fare'] == 0
ALONE = train_df['SibSp'] == 0
NOTALONE = train_df['SibSp'] > 0
ALLCLASS = train_df['Pclass']
C1 = train_df['Pclass'] == 1
C2 = train_df['Pclass'] == 2
C3 = train_df['Pclass'] == 3
SM = train_df['Sex'] =='male'
SF = train_df['Sex'] == 'female'
CHILD1 = train_df['Age'] < 15 & C1
CHILD2 = train_df['Age'] < 15 & C2
CHILD3 = train_df['Age'] < 15 & C3
FCHILD = CHILD & SF
MCHILD = CHILD & SM

#pclassgraph = sns.barplot(train_df['Pclass'], train_df['Survived']*100, hue = train_df['Sex'])
#plt.show()
#pclassgraph.set(xlabel='Passenger Class', ylabel='Percent Survived')
#plt.show()

#childrengraph = sns.barplot(train_df['Pclass'], train_df['Survived']*100, hue = Children)
#childrengraph.set(xlabel ='Passenger Class', ylabel='Percent Survived')
#plt.show()

siblingsgraph = sns.barplot(train_df['Pclass'], train_df['Survived'], hue = train_df['SibSp'])
plt.show()

femalealonegraph = sns.barplot(train_df['Pclass'], train_df['Survived'], hue = ALONE & SF)
plt.show()

femalenotalonegraph = sns.barplot(train_df['Pclass'], train_df['Survived'], hue = NOTALONE & SF)
plt.show()
pred_survived_train = SF
print(pd.crosstab(pred_survived_train, train_df['Survived']))
print(np.mean(pred_survived_train ==  train_df['Survived']))

pred_survived_train2 = ((C1 | C2) & SF) | (CHILD == True)
print(pd.crosstab(pred_survived_train2, train_df['Survived']))
print(np.mean(pred_survived_train2 == train_df['Survived'])*100)

pred_survived_train3 = ((C1 | C2) & SF) | (CHILD == True)
print(pd.crosstab(pred_survived_train3, train_df['Survived']))
print(np.mean(pred_survived_train3 == train_df['Survived'])*100)

pred_survived_train4 = ((C1 | C2) & SF) | (CHILD == True) | (C3 & ALONE & SF)
print(pd.crosstab(pred_survived_train4, train_df['Survived']))
print(np.mean(pred_survived_train4 == train_df['Survived'])*100)



test_df['Survived'] = (pred_survived_train4).astype(int)
submission = test_df[['PassengerId', 'Survived']] 

submission.to_csv("sub5.csv", index = None)