# Loading the necessary libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

import csv as csv 



# Input data files are available in the "../input/" directory.

# For .read_csv, always use header=0 when you know row 0 is the header row

train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)
# Taking a first look at the data

train_df.head()
test_df.head()
# Concatenate the training and testing datasets, to facilitate feature engineering.

# After this we can do our engineering on only a single dataframe.

df = pd.concat([train_df, test_df], axis=0)
# First we map the "Sex" category into a "Gender" category, being "0" for female and "1" for male 

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df = df.drop(['Sex'], axis=1) #remove the original "Sex" column
# 1-hot encoding for the Port of Embarkation

df['Cherbourg'] = 0

df['Queenstown'] = 0

df['Southampton'] = 0

df.loc[df.Embarked == 'C', 'Cherbourg'] = 1

df.loc[df.Embarked == 'Q', 'Queenstown'] = 1

df.loc[df.Embarked == 'S', 'Southampton'] = 1
# Add family size 

df['FamilySize'] = df['SibSp'] + df['Parch']
median_ages = np.zeros((2,3))

for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & \

                              (df['Pclass'] == j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']

for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]
# The original "Age" column can now be removed

df = df.drop(['Age'], axis=1)
# There is 1 missing Fare value, which we replace with the median of all fares

df.loc[df.Fare.isnull(),'Fare'] = df.Fare.dropna().median()
# Add a column indicating whether (1) or not (0) a cabin was assigned

df['CabinAssigned'] = (df.Cabin.isnull() != 1).astype(int)
# Now we can remove the columns that we will not use for the regression model

df = df.drop(['Cabin', 'Embarked', 'Name', 'Ticket'],axis=1)
# We store the survival status of the training set in a separate array, needed to train the model later

survived = df[df['Survived'].isnull() == False]['Survived'].values



train_final = df[df['Survived'].isnull() == False]

train_final = train_final.drop(['Survived'],axis=1).values

test_final = df[df['Survived'].isnull() == True]

test_final = test_final.drop(['Survived'],axis=1).values
logistic_regression = LogisticRegression()

logistic_regression.fit(train_final,survived)

output = logistic_regression.predict(test_final)
# Create a dataframe of the output, making it easier to save as a .csv file

output_df = pd.DataFrame({'PassengerId':df[df['Survived'].isnull() == True].PassengerId, 'Survived':output.astype(int)})

output_df.head()
output_df.to_csv('logisticregressionmodel.csv', header=True)