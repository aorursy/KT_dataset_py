# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Inputing the train and test data that will be used

titanic_train = pd.read_csv('../input/train.csv')

titanic_test = pd.read_csv('../input/test.csv')
#Data preprocessing before used in a neural network

titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())

titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())

titanic_train['Fare'] = titanic_train['Fare'].fillna(titanic_train['Fare'].median())

titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())

titanic_train.loc[titanic_train['Sex'] == 'male', 'Sex'] = 0

titanic_train.loc[titanic_train['Sex'] == 'female', 'Sex'] = 1

titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0

titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1

clf_train_output = titanic_train['Survived'].values.tolist()

clf_train_input = titanic_train[['Pclass','Sex','Age','SibSp','Parch','Fare']]

clf_test_input = titanic_test[['Pclass','Sex','Age','SibSp','Parch','Fare']]
#Random Forest Classification

random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(clf_train_input,clf_train_output)
#Running Random Forest Classifier on test dataset

random_forest_output = random_forest.predict(clf_test_input)
#Calculating accuracy of Random Forest classification

random_forest_accuracy = random_forest.score(clf_train_input,clf_train_output) * 100

random_forest_accuracy
#defining which output to use for submission

output = random_forest_output
#Submitting result

submission = pd.DataFrame({

    "PassengerId": titanic_test["PassengerId"],

    "Survived": output

})

submission.to_csv('submission.csv', index = False)

submission