# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.model_selection import cross_val_score

# Any results you write to the current directory are saved as output.

import seaborn as sns
# read the Dataset

dataset = pd.read_csv('../input/train.csv')

dataset.head()

actual_data = dataset

actual_data
# Different datatypes

dataset.dtypes
# Now we train differnt Machine Learning Models to get insights from data

# removing unncessary data since name of the passanger plays no signifiant role

def clean_data(dataset):

    #removes all the nan values

    data = dataset.dropna()

    #drop the passenger id

    data  = data.drop('PassengerId',axis=1)

    #now drop  the name feild from the coloumns

    data = data.drop('Name',axis=1)

    #dropping the Tiket from the Feature

    data = data.drop('Ticket',axis=1)

    #now we drop

    return data

    
# encodes the Categorical Features 

def encode_categorical_feautures(dataset):

    for coloumns in dataset.columns:

        enc = LabelEncoder()

        enc.fit(dataset[coloumns])

        dataset[coloumns] = enc.fit_transform(dataset[coloumns])

    return dataset    
dataset = clean_data(dataset)

dataset = encode_categorical_feautures(dataset)
# try to visualize gender and survival analysis

survived = dataset[dataset['Survived'] == 1]

survived_males = survived[survived['Sex'] == 0]

survived_females = survived[survived['Sex'] == 1]

print("Out of Total Passengers ",len(dataset))

print("Males Survived :",len(survived_males))

print("Females Survived :",len(survived_females))

sns.set_style("whitegrid")

sns.barplot(x=["Males","Females"],y=[len(survived_males),len(survived_females)])
# Now lets analyze which class of people were more likely to Die

classes = actual_data[actual_data['Survived'] == 0]

# classes

classes = classes.groupby(['Pclass'])['Pclass'].count()

sns.barplot(x=[1,2,3],y= classes)
# Now finally the whole Dataset and categorical Features encoded

dataset
# using training and test data

train_y = dataset['Survived']

train_x = dataset.drop('Survived',axis=1)
# cleaning and encode categorical features

test_data = pd.read_csv('../input/test.csv')

test_data = clean_data(test_data)

test_data = encode_categorical_feautures(test_data)

test_x = test_data
clf = GaussianNB()

clf.fit(train_x,train_y)

#y = clf.predict(test_x)

scores = cross_val_score(clf, train_x, train_y, cv=5)

# Prints the cross validation score 

print("Naive Bayes Classifier :\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf.fit(train_x,train_y)

scores = cross_val_score(clf,train_x,train_y,cv=5)

print("Decision Tree Classifier :\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)

clf.fit(train_x,train_y)

scores = cross_val_score(clf,train_x,train_y,cv = 5)

print("Random Forest Classifier :\n")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))