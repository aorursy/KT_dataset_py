#Load the packages that we will use

import pandas as pd

import numpy as np

import csv as csv

from sklearn import ensemble

from sklearn import tree
#Finding the working directory

import os

os.getcwd()
#Check what files are in the working directory

from subprocess import check_output

print (check_output(["ls", "../working"]).decode("utf8"))
#Check what files are in the working directory

from subprocess import check_output

print (check_output(["ls", "../input"]).decode("utf8"))
#Change it if not convenient

os.chdir('/kaggle/input')



#Verify it has been changed successfully

os.getcwd()
train_df = pd.read_csv('train.csv', header=0)
whos
#Count number of rows and columns

train_df.shape
#Get information about the variables in the dataframe

train_df.info()
train_df.describe().transpose()
# female = 0, male = 1

train_df['Gender']=train_df['Sex'].map({'female':0,'male':1}).astype(int)
train_df.info()
#all ages with no data - make median

median_age=train_df['Age'].dropna().median()

if len(train_df.Age[train_df.Age.isnull()]) > 0:

    train_df.loc[(train_df.Age.isnull()),'Age'] = median_age
train_df.info()
# All missing Embarked -> just make them embark from most common place

mode_embark = train_df['Embarked'].dropna().mode().values





if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:

    train_df.loc[ (train_df.Embarked.isnull()),'Embarked' ] = mode_embark
# Embarked from 'C', 'Q', 'S'





Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,

Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
# All the missing Fares -> assume median of their respective class

if len(train_df.Fare[ train_df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = train_df[ train_df.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        train_df.loc[ (train_df.Fare.isnull()) & (train_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
# Data cleanup

# TEST DATA

test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe





# I need to do the same with the test data now, so that the columns are the same as the training data

# I need to convert all strings to integer classifiers:

# female = 0, Male = 1

test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)





# All the ages with no data -> make the median of all Ages

median_age = test_df['Age'].dropna().median()

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:

    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

    

# All missing Embarked -> just make them embark from most common place

mode_embark = test_df['Embarked'].dropna().mode().values

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

    test_df.loc[ (test_df.Embarked.isnull()),'Embarked' ] = mode_embark





# Again convert all Embarked strings to int

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)





# All the missing Fares -> assume median of their respective class

if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]





# Collect the test data's PassengerIds before dropping it

ids = test_df['PassengerId'].values

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
train_data = train_df.values

test_data = test_df.values
whos
print ('Training...')

forest = ensemble.RandomForestClassifier(n_estimators=100)

forest = forest.fit( train_data[0::,1::], train_data[0::,0] )





print ('Predicting...')

output = forest.predict(test_data).astype(int)









predictions_file = pd.DataFrame({'PassengerId':ids, 'Survived':output})

print ('Done.')
import matplotlib.pyplot as plt