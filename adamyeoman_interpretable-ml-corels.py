# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install corels



# https://pycorels.readthedocs.io/en/latest/CorelsClassifier.html

# https://pycorels.readthedocs.io/en/latest/examples.html
from corels import *
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head()
y = train_df.pop('Survived')

X = train_df
def feature_engineering(X):

    X['isFemale'] = X['Sex'].replace({'male': 0, 'female': 1})

    

    X['Pclass1'] = 1*(X['Pclass']==1)

    X['Pclass2'] = 1*(X['Pclass']==2)

    X['Pclass3'] = 1*(X['Pclass']==3)



    X['has1SiblingOrSpouse'] = 1*(X['SibSp'] == 1)

    X['hasMoreThan1SiblingOrSpouse'] = 1*(X['SibSp'] > 1)



    X['has1ParentOrChild'] = 1*(X['Parch'] == 1)

    X['hasMoreThan1ParentOrChild'] = 1*(X['Parch'] > 1)

    

    X['fareAboveMean'] = 1*(X['Fare'] > X['Fare'].mean())

    X['fareAboveMedian'] = 1*(X['Fare'] > X['Fare'].median())

    

    X['EmbarkedC'] = 1*(X['Embarked']=='C')

    X['EmbarkedS'] = 1*(X['Embarked']=='S')

    X['EmbarkedQ'] = 1*(X['Embarked']=='Q')

    

    X['isUnder18Yrs'] = 1*(X['Age'] < 18)

    X['isOver60Yrs'] = 1*(X['Age'] > 60)

    

    return X
X = feature_engineering(X)
X.head()
features = ['isFemale',

            'Pclass1','Pclass2','Pclass3',

            'has1SiblingOrSpouse','hasMoreThan1SiblingOrSpouse',

            'has1ParentOrChild','hasMoreThan1ParentOrChild',

            'fareAboveMean','fareAboveMedian',

            'EmbarkedC','EmbarkedS','EmbarkedQ',

            'isUnder18Yrs','isOver60Yrs']

X_train = X[features]
# Create the model, with 10000 as the maximum number of iterations 

c = CorelsClassifier(n_iter=10000, 

                     max_card=3, # How many features can we combine in each statement?

                     c = 0.001 # Higher values penalise longer rulelists

                    )



# Fit, and score the model on the training set

c.fit(X_train, y, features=features, prediction_name="Survived")

a = c.score(X_train, y)



# Print the model's accuracy on the training set

print(a)
# Predict for the test data

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df.head()
X_test = feature_engineering(test_df)
X_test = X_test[features]
predictions = c.predict(X_test)
submission = pd.DataFrame({'PassengerID':test_df['PassengerId'],

                           'Survived':1*predictions})



submission.to_csv("submission.csv", index=False)
submission

