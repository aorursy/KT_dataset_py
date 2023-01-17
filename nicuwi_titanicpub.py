'''This is a test and sandbox sketch'''



import pandas as pd

import numpy as np

from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import cross_val_score

#!pip install decision-tree-id3

#!pip install pydotplus

#from id3 import Id3Estimator

#from id3 import export_graphviz

#from id3 import export_text

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



#=============Analyze data============

#1. Drop Ticket since this feature is just alphanumeric values that dont relate to the output

#   Drop Cabin for the same reason. Even though cabin (code) might be shared by more pearsons this relationship can be extracted from SibSp and ParCh

train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)



#2. Name can be dropped as it does not converge any relevance

train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)



#3. Combine the datasets for ease of use further

combine = [train, test]



#4. Map male and female to numericals

for dataset in combine:  

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



#5. Add the median for Age

guess_ages = np.zeros((2,3))

for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



#6. Create age groups

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']



#7. Map embark port to numerical. Fill NA with the most common port.

freq_port = train.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

#8. Fill the missing fares with the median and create fare groups

for dataset in combine:

    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

   

X_train = combine[0].drop("Survived", axis=1)

X_train = X_train.drop("PassengerId", axis=1)

Y_train = combine[0]["Survived"]

X_test  = combine[1].drop("PassengerId", axis=1).copy()



#=============Run algo============

#cart

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO                                                                    



cart = DecisionTreeClassifier()

cart.fit(X_train, Y_train, check_input = True)



#for cross validate, use only train since test does not provide 'Surived' column

full_df = combine[0]

y = full_df.Survived



X =  pd.DataFrame(full_df, columns=['Age', 'Embarked', 'Fare', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp'])



#cross validate

results = []



cart_cross = cross_val_score(cart, X, y, cv=5, scoring='accuracy')





print("Accuracy CART: %0.2f (+/- %0.2f)" % (cart_cross.mean(), cart_cross.std() * 2))



Y_test = cart.predict(X_test)



combine[1].drop(combine[1].columns.difference(['PassengerId']), 1, inplace=True)

combine[1]["Survived"] = Y_test

print(combine[1].head(10))



#submit file



submit = combine[1][['PassengerId','Survived']]

submit.to_csv("../working/submission.csv", index=False)
