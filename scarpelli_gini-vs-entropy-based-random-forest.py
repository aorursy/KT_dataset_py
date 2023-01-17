# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Imports

import numpy as np

import pandas as pd

import csv as csv

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

%matplotlib inline
#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#Print to standard output, and see the results in the "log" section below after running your script

print("\n\nTop of the training data:")

print(train.head())



print("\n\nSummary statistics of training data")

print(train.describe())
# This part is from AstroDave's example, I've made no changes here on the Data cleanup.





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

ids = test['PassengerId'].values

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 





# The data is now ready to go. So lets fit to the train, then predict to the test!

# Convert back to a numpy array

train_data = train_df.values

test_data = test_df.values
# We are going to test from 10 trees to 149 trees in the random forest

n_test = 140

gini = np.zeros(n_test) # to store the score values on the training set, will use cross validation next time

entropy = np.zeros(n_test) # to store the score values on the training set, will use cross validation next time



print("Training using Gini as impurity function")

for i in range(0,n_test):

    temp = 10 + i

    forest = RandomForestClassifier(n_estimators= temp)

    forest = forest.fit( train_data[0::,1::], train_data[0::,0])

    gini[i] = forest.score(train_data[0::,1::], train_data[0::,0]) # as stated above, I'm calculating the score on the training set at the moment, I'll try a Cross Validation next time



print("Training using Entropy as impurity function")

for i in range(0,n_test):

    temp = 10 + i

    forest = RandomForestClassifier(n_estimators= temp, criterion = 'entropy')

    forest = forest.fit( train_data[0::,1::], train_data[0::,0])

    entropy[i] = forest.score(train_data[0::,1::], train_data[0::,0]) # as stated above, I'm calculating the score on the training set at the moment, I'll try a Cross Validation next time
ab = range(10,150)

plt.figure(figsize=(20,20))

l_gini, = plt.plot(ab,gini,'b')

l_entropy, = plt.plot(ab,entropy,'r')

plt.legend([l_gini, l_entropy], ['Gini', 'Entropy'])