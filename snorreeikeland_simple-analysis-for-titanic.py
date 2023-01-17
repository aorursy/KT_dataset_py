import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#import csv-files and merge the sets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

merged = train.append(test)

merged.info()
merged.head()
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

plt.show()

train[['Sex','Survived']].groupby(['Sex']).mean()
#Extract title and group them

Title=[]

for i in range(len(merged)):

    names = merged.Name.values[i].replace('.',',').split(', ')

    title = names[1]

    Title.append(title)

merged['PTitle'] = Title



pd.crosstab(merged['PTitle'],merged['Sex'])
merged['PTitle'] = merged['PTitle'].replace(['Lady', 'the Countess','Mlle', 'Ms', 'Mme', 'Dona','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

merged[['Sex','PTitle','Pclass','Survived']].groupby(['Sex','PTitle','Pclass']).mean()
#Add family size groups

merged["FamilySize"] = merged["SibSp"]+merged["Parch"]+1

merged[['FamilySize','Survived']].groupby(['FamilySize']).mean().plot.bar()

plt.show()
merged.loc[merged['FamilySize'] == 1, 'Fsize'] = 'Alone'

merged.loc[(merged['FamilySize'] > 1)  &  (merged['FamilySize'] < 5) , 'Fsize'] = 'Small'

merged.loc[merged['FamilySize'] >4, 'Fsize'] = 'Large'

fem_analysis = merged[merged['Sex'] == 'female']

fem_analysis[['PTitle','Pclass','Survived','Fsize']].groupby(['Pclass','PTitle','Fsize']).mean()
fem2_analysis = fem_analysis[ (fem_analysis['Pclass'] == 3) &

                              (fem_analysis['Fsize'] != 'Large')]

fem2_analysis[['PTitle','Embarked','Survived']].groupby(['PTitle','Embarked']).mean()
fem2_analysis[['PTitle','Embarked','Survived']].groupby(['PTitle','Embarked']).count()
men_analysis = merged[merged['Sex'] == 'male']

men_analysis[['PTitle','Pclass','Survived','Fsize']].groupby(['PTitle','Pclass','Fsize']).mean()
men2_analysis = men_analysis[(men_analysis['Pclass'] == 1) &

                             (men_analysis['PTitle'] != 'Master') &

                             (men_analysis['Fsize'] != 'Large')]

men2_analysis['HasCabin'] = men2_analysis['Cabin'].notnull()

men2_analysis[['PTitle','HasCabin','Survived']].groupby(['PTitle','HasCabin']).mean()
merged['Group'] = merged['Sex']



#Females who don't survive

merged.loc[(merged['Sex'] == 'female') & 

           (merged['Fsize'] == 'Large') &

           (merged['Pclass'] == 3), 'Group'] = 'Females, large family, class 3'



merged.loc[(merged['PTitle'] == 'Miss') & 

           (merged['Fsize'] != 'Large') &

           (merged['Pclass'] == 3) &

           (merged['Embarked'] == 'S'), 'Group'] = 'Miss, class 3, embarked Southampton'



#Males who survive

merged.loc[(merged['PTitle'] == 'Master') & 

           ((merged['Pclass'] < 3) | 

            (merged['Fsize'] == 'Small')), 'Group'] = 'Masters in Pclass 1&2 or small families'



merged.loc[(merged['PTitle'] == 'Rare') & 

           (merged['Sex'] == 'male') & 

           (merged['Pclass'] == 1) &

           (merged['Cabin'].notnull()) &

           (merged['Fsize'] != 'Large'), 'Group'] = 'Rare title, Cabin Pclass 1, not large'



merged[['Group','Survived']].groupby(['Group'], sort=False).mean().plot.bar()

plt.show()

merged[['Group','Survived']].groupby(['Group'], sort=False).mean()
merged['Predict'] = 1

merged.loc[merged['Group'] == 'male', 'Predict'] = 0

merged.loc[merged['Group'] == 'Females, large family, class 3', 'Predict'] = 0

merged.loc[merged['Group'] == 'Miss, class 3, embarked Southampton', 'Predict'] = 0

merged.loc[merged['Group'] == 'Masters in Pclass 1&2 or small families', 'Predict'] = 1

merged.loc[merged['Group'] == 'Rare title, Cabin Pclass 1, not large', 'Predict'] = 1



#Score accuracy on training set

train_set = merged[merged['Survived'].notnull()]

train_set['Score'] = train_set['Survived'] == train_set['Predict']

print("Accuracy training set:")

print(train_set['Score'].mean())
#select test set          

test_analysed = merged[merged['Survived'].isnull()]



#See how it predicts various titles will do

test_analysed[['PTitle','Predict']].groupby(['PTitle']).mean().plot.bar()

plt.show()

test_analysed[['Sex','PTitle','Predict']].groupby(['Sex','PTitle']).mean()
#Set solution output

my_solution = pd.DataFrame({'PassengerId': test_analysed['PassengerId'],

                            'Survived':test_analysed['Predict']})

my_solution.to_csv('submission.csv', index = False)