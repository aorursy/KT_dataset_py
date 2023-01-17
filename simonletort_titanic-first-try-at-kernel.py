# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv

import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# TRAINING DATASET

train_df = pd.read_csv('../input/train.csv', header=0)        # Load the train file into a dataframe

# TESTING DATASET

test_df = pd.read_csv('../input/test.csv', header=0)        # Load the test file into a dataframe

# FULL DATASET

full_df = pd.concat([train_df, test_df],keys = ['train','test'])
train_df.head()
train_df.describe()
test_df.describe()
full_df.describe()
# how many people in same cabin?

full_df.groupby('Cabin').count()
# Data cleanup





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

#train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
train_df


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

#test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin'], axis=1) 
test_df
X = train_data[0::,1::]

y = train_data[0::,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# The data is now ready to go. So lets fit to the train, then predict to the test!

# Convert back to a numpy array

train_data = train_df.values

test_data = test_df.values





#print 'Training...'

forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit(X, y)



#print 'Predicting...'

output = forest.predict(test_data).astype(int)



forest.score(train_data[0::,1::], train_data[0::,0])
output
predictions_file = open("myfirstforest.csv", "wb")

open_file_object = csv.writer(predictions_file)

open_file_object.writerow(["PassengerId","Survived"])

open_file_object.writerows(zip(ids, output))

predictions_file.close()
X = train_data[0::,1::]

y = train_data[0::,0]



# instantiate a logistic regression model, and fit with X and y

model = LogisticRegression()

model = model.fit(X , y)



# check the accuracy on the training set

model.score(X, y)
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):

    plt.figure(figsize=(10,6))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel(scoring)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,

                                                            n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
def get_model(estimator, parameters, X_train, y_train, scoring):  

    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)

    model.fit(X_train, y_train)

    return model.best_estimator_
from sklearn.metrics import accuracy_score

scoring = make_scorer(accuracy_score, greater_is_better=True)
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(random_state=42, penalty='l1')

parameters = {'C':[0.5]}

clf_lg1 = get_model(lg, parameters, X, y, scoring)

print (clf_lg1)

print (accuracy_score(y_test, clf_lg1.predict(X_test)))

plot_learning_curve(clf_lg1, 'Logistic Regression', X, y, cv=4);
y_ = clf_lg1.predict(test_df)

# Append the survived attribute to the test data



#test_df['Survived'] = y_

#predictions = test_df[['PassengerId', 'Survived']]

#print(predictions)



# Save the output for submission

#predictions.to_csv('submission.csv', index=False)
print(y_.size)

print(test_df.size)