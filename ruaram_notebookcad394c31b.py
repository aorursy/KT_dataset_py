# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df_train = pd.read_csv('../input/train.csv', header=0)

df_test = pd.read_csv('../input/test.csv', header=0)



## First of all, it's hard to run analysis on the string values of "male" and "female". 

## Let's practice transforming Sex data and store our transformation in a new column, so the original Sex isn't changed.

## df['Gender'] = df['Sex'].map( lambda x: x[0].upper() ) 

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



## let's use the age that was typical in each passenger class for the missing values of Age. 

## Start by building another reference table to calculate what each of these medians are:

median_ages = np.zeros((2,3))



for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()



median_ages



## Make a copy of Age:



df['AgeFill'] = df['Age']



for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]

        

##Let's also create a feature that records whether the Age was originally missing. 

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)



## Since we know that Parch is the number of parents or children onboard, 

## and SibSp is the number of siblings or spouses, we could collect those together as a FamilySize

df['FamilySize'] = df['SibSp'] + df['Parch']





## We can also create artificial features if we think it 

## may be advantageous to a machine learning algorithm 

df['Age*Class'] = df.AgeFill * df.Pclass



## Create new column

df['Embarked_t'] = df['Embarked'] 



most_common = df['Embarked_t'].dropna().mode()

most_common.to_string()



df['Embarked_t'] = df['Embarked_t'].fillna('S') 



## Let's also transform embarked data

df['Embarked_t'] = df['Embarked_t'].map( {'C': 0, 'Q': 1, 'S':2} ).astype(int)
## let's take a count of the males in each class.

for i in range(1,4):

    print('Class {:d}, No. of males: {:d}'.format(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])))
import pylab as P

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)

P.show()
## let's use the age that was typical in each passenger class for the missing values of Age. 

## Start by building another reference table to calculate what each of these medians are:

median_ages = np.zeros((2,3))



for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()



median_ages
df.describe()

##determine what columns we have left which are not numeric



df.dtypes[df.dtypes.map(lambda x: x=='object')]
## The next step is to drop the columns which we will not use:



df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 
##The final step is to convert it into a Numpy array. 

## Pandas can always send back an array using the .values method. Assign to a new variable, train_data:



train_data = df.values

train_data
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)



# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])



# Take the same decision trees and run it on the test data

output = forest.predict(test_data)