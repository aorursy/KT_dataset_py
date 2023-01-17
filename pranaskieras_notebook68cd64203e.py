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



import tensorflow as tf



train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")

test_set



train_labels = train_set['Survived']

test_labels = []



import os

for file in os.listdir("../input"):

    print(os.path.join("../input", file))

        

train_set.drop('Survived', axis=1, inplace=True)



#data exploration



#droping the id as it is not relevant

train_set.drop('PassengerId', axis=1, inplace=True)

test_set.drop('PassengerId', axis=1, inplace=True)

#convert sex into  0 and 1

train_set['Sex'] = train_set['Sex'].map({'male':1, 'female':0})

test_set['Sex'] = test_set['Sex'].map({'male':1, 'female':0})



#parse name to get title

train_set['Name'] = train_set['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

test_set['Name'] = test_set['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

titles = {

    "Officer":0,

    "Royalty":1,

    "Mrs":2,

    "Miss":3,

    "Mr":4,

    "Master": 5

}

Title_Dictionary = {

                        "Capt":       titles["Officer"],

                        "Col":        titles["Officer"],

                        "Major":      titles["Officer"],

                        "Jonkheer":   titles["Royalty"],

                        "Don":        titles["Royalty"],

                        "Sir" :       titles["Royalty"],

                        "Dr":         titles["Officer"],

                        "Rev":        titles["Officer"],

                        "the Countess":titles["Royalty"],

                        "Dona":       titles["Royalty"],

                        "Mme":        titles["Mrs"],

                        "Mlle":       titles["Miss"],

                        "Ms":         titles["Mrs"],

                        "Mr" :        titles["Mr"],

                        "Mrs" :       titles["Mrs"],

                        "Miss" :      titles["Miss"],

                        "Master" :    titles["Master"],

                        "Lady" :      titles["Royalty"]



                        }

train_set['Name']=train_set.Name.map(Title_Dictionary)

test_set['Name']=test_set.Name.map(Title_Dictionary)

#scale the numeric variables to not distort the distribution

ages = train_set['Age'].fillna(0)

average_age = np.average(ages)

train_set['Age'] = train_set['Age'].fillna(average_age)

train_set['Age'] /= np.max(train_set['Age'], axis=0)



ages = test_set['Age'].fillna(0)

average_age = np.average(ages)

test_set['Age'] = test_set['Age'].fillna(average_age)

test_set['Age'] /= np.max(test_set['Age'], axis=0)





fares = train_set['Fare'].fillna(0)

average_fare = np.average(fares)

train_set['Fare'] = train_set['Fare'].fillna(average_fare)

train_set['Fare'] /= np.max(train_set['Fare'], axis=0)

average_fare = np.average(fares)



fares = test_set['Fare'].fillna(0)

average_fare = np.average(fares)

test_set['Fare'] = test_set['Fare'].fillna(average_fare)

test_set['Fare'] /= np.max(test_set['Fare'], axis=0)

average_fare = np.average(fares)



#remove the cabin, ticket number and port of embarkation as it does not represent any new information

train_set.drop('Cabin', axis=1, inplace=True)

train_set.drop('Ticket', axis=1, inplace=True)

train_set.drop('Embarked', axis=1, inplace=True)



test_set.drop('Cabin', axis=1, inplace=True)

test_set.drop('Ticket', axis=1, inplace=True)

test_set.drop('Embarked', axis=1, inplace=True)





pd.set_option('display.max_rows', 1000)

#print(test_set)

#tf.contrib.learn.LinearClassifier(feature_columns=)



from sklearn.linear_model import LogisticRegression



learn = LogisticRegression()

learn.fit(train_set, train_labels)



pred = learn.predict(test_set)

acc_log = round(learn.score(train_set, train_labels) * 100, 2)

pred