# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# female = 0, Male = 1

train_df = pd.read_csv('../input/train.csv', header=0)

train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# All missing Embarked -> just make them embark from most common place

if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:

    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values



#Encoding the embarked labels

le = LabelEncoder()

le.fit_transform(train_df['Embarked'].unique())



train_df['Embarked'] = le.transform(train_df['Embarked'])
# All the ages with no data -> make the median of all Ages

median_age = train_df['Age'].dropna().median()

if len(train_df.Age[ train_df.Age.isnull() ]) > 0:

    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age



# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
# TEST DATA

test_df = pd.read_csv('../input/test.csv', header=0)



test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# All missing Embarked -> just make them embark from most common place

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values





le.fit_transform(test_df['Embarked'].unique())



test_df['Embarked'] = le.transform(test_df['Embarked'])



# All the ages with no data -> make the median of all Ages

median_age = test_df['Age'].dropna().median()

if len(test_df.Age[ test_df.Age.isnull() ]) > 0:

    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age



ids = test_df['PassengerId'].values



# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)



# All the missing Fares -> assume median of their respective class

if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:

    median_fare = np.zeros(3)

    for f in range(0,3):                                              # loop 0 to 2

        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()

    for f in range(0,3):                                              # loop 0 to 2

        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]
# The data is now ready to go. So lets fit to the train, then predict to the test!

# Convert back to a numpy array

train_data = train_df.values

test_data = test_df.values





print('Training...')

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit( train_data[0::,1::], train_data[0::,0] )



print('Predicting...')

output = forest.predict(test_data).astype(int)



print('Score:', forest.score(train_data[0::,1::], train_data[0::,0]))
predictions_file = open("myfirstforest.csv", "w")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(ids, output))

predictions_file.close()

print('Done.')