# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
url = '../input/train.csv'



data = pd.read_csv(url, index_col = 'PassengerId')

data.head(3)

def Extract_Title(Words):

    for word in Words.split():

        if word.lower() == 'mr.':

            return int(0)

        elif word.lower() == 'mrs.':

            return int(1)

        elif (word.lower()) == 'miss.' or (word.lower() == 'ms.'):

            return int(2)

        elif word.lower() == 'master.':

            return int(3)

        elif word.lower() == 'don.':

            return int(4)

        elif word.lower() == 'rev.':

            return int(5)

        elif word.lower() == 'dr.':

            return int(6)

        elif word.lower() == 'mme.':

            return int(7)

        elif word.lower() == 'major.':

            return int(8)

        elif word.lower() == 'lady.':

            return int(9)

        elif word.lower() == 'sir.':

            return int(10)

        elif word.lower() == 'col.':

            return int(11)

        elif word.lower() == 'mlle.':

            return int(12)

        elif word.lower() == 'capt.':

            return int(13)

        elif word.lower() == 'countess.':

            return int(14)

        elif word.lower() == 'jonkheer.':

            return int(15)

        elif word.lower() == 'dona.':

            return int(16)

    

    return 'none'



def Shave(text):

    if str(text) != 'nan':

        return text[0]

    else:

        return 'unknown'

    

def Cabin_Numerize(x):

    if x == 'unknown':

        return 0

    elif x == 'A':

        return 1

    elif x == 'B':

        return 2

    elif x == 'C':

        return 3

    elif x == 'D':

        return 4

    elif x == 'E':

        return 5

    elif x == 'F':

        return 6

    elif x == 'G':

        return 7

    elif x == 'T':

        return 8

    

def Sex_Numerize(x):

    if x == 'male':

        return 0

    elif x == 'female':

        return 1

    

def Embarked_Numerize(x):

    y = str(x)

    

    if y == 'S':

        return 0

    if y == 'C':

        return 1

    if y == 'Q':

        return 2

    

def Round_Predictions(x):

    

    if x <= 0.5:

        return 0

    if x > 0.5:

        return 1

    





data['Cabin'] = data['Cabin'].astype(str)

data['Cabin_Level'] = data['Cabin'].apply(Shave)

data['Cabin_Level'] = data['Cabin_Level'].apply(Cabin_Numerize)

data['Sex'] = data['Sex'].apply(Sex_Numerize)

data['Embarked'] = data['Embarked'].apply(Embarked_Numerize)

data['Title'] = data['Name'].apply(Extract_Title)



data = data.reindex(columns=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',

              'Embarked', 'Title', 'Cabin_Level'])



y_train = data['Survived']

X_train = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',

              'Embarked', 'Title', 'Cabin_Level']]
my_model = XGBRegressor()

my_model.fit(X_train, y_train)
test_url = '../input/test.csv'

test_data = pd.read_csv(test_url)
test_data['Cabin'] = test_data['Cabin'].astype(str)

test_data['Cabin_Level'] = test_data['Cabin'].apply(Shave)

test_data['Cabin_Level'] = test_data['Cabin_Level'].apply(Cabin_Numerize)

test_data['Sex'] = test_data['Sex'].apply(Sex_Numerize)

test_data['Embarked'] = test_data['Embarked'].apply(Embarked_Numerize)

test_data['Title'] = test_data['Name'].apply(Extract_Title)



test_data = test_data.reindex(columns=['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',

              'Embarked', 'Title', 'Cabin_Level'])



X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',

              'Embarked', 'Title', 'Cabin_Level']]

titanic_predictions = my_model.predict(X_test)

titanic_submission_1 = pd.DataFrame()

titanic_submission_1['PassengerId'] = test_data['PassengerId']

titanic_submission_1['Survived'] = titanic_predictions

titanic_submission_1['Survived'] = titanic_submission_1['Survived'].apply(Round_Predictions)

titanic_submission_1
titanic_submission_1.to_csv('titanic_submission_1.csv', index=False)