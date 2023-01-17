# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import xgboost as xgb

import csv as csv

from sklearn.ensemble import RandomForestClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../working"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
""" Writing my first randomforest code.

Author : AstroDave Edited by Joshua Herman

Date : 23rd September 2012

Revised: 15 April 2014

please see packages.python.org/milk/randomforests.html for more



""" 





# Data cleanup

# TRAIN DATA

train_df = pd.read_csv('../input/train.csv', header=0)        # Load the train file into a dataframe



# I need to convert all strings to integer classifiers.

# I need to fill in the missing values of the data and make it complete.



# female = 0, Male = 1

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Embarked from 'C', 'Q', 'S'

# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.



# All missing Embarked -> just make them embark from most common place

if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:

    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values



Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,

Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int



# All the ages with no data -> make the median of all Ages

median_age = train_df['Age'].dropna().median()

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:

    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age



# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 





# TEST DATA

test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe



# I need to do the same with the test data now, so that the columns are the same as the training data

# I need to convert all strings to integer classifiers:

# female = 0, Male = 1

test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# Embarked from 'C', 'Q', 'S'

# All missing Embarked -> just make them embark from most common place

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

# Again convert all Embarked strings to int

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)





# All the ages with no data -> make the median of all Ages

median_age = test_df['Age'].dropna().median()

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:

    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age



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





# The data is now ready to go. So lets fit to the train, then predict to the test!

# Convert back to a numpy array

train_data = train_df.values

test_data = test_df.values

dtrain = xgb.DMatrix(train_df.values)

dtest = xgb.DMatrix(test_df.values)



'''print('Training...')

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( train_data[0::,1::], train_data[0::,0] )



print('Predicting...')

#output = forest.predict(test_data).astype(int)





predictions_file = open("myfirstforest.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(ids, output))

predictions_file.close()

print('Done.')'''
print(check_output(["ls", "../working"]).decode("utf8"))
from sklearn.grid_search import GridSearchCV

param = {'silent':1, 'objective':'reg:gamma', 'booster':'gbtree', 'base_score':3}



# the rest of settings are the same



watchlist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 30

xgb_model = xgb.XGBClassifier()

clf = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'n_estimators': [50,100,200]}, verbose=1)

clf.fit(train_data[0::,1::], train_data[0::,0] )

print('Predicting...')

output = clf.predict(test_data).astype(int)

#xgb.predict_proba(test)

#bst = xgb_model.train(param, dtrain, num_round, watchlist)

#preds = output.predict(dtest)

labels = dtest.get_label()



#print ('test deviance=%f' % (2 * np.sum((labels - preds) / preds - np.log(labels) + np.log(preds))))
predictions_file = open("myfirstxgb.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(ids, output))

predictions_file.close()

print('Done.')
print(output)
print(labels)