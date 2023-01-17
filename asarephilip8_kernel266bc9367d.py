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
%ls
test = pd.read_csv('/kaggle/input/titanic/test.csv').copy()

train = pd.read_csv('/kaggle/input/titanic/train.csv').copy()

sub =  pd.read_csv('/kaggle/input/titanic/gender_submission.csv').copy()
print(test.columns.sort, '\n')

print(train.columns.sort)
train.head(5)
test.head(5)
#Survived men

survived_men = train.loc[train.Sex == 'male']['Survived']

print( '% men who survived = ', (sum(survived_men)/ len(survived_men)) * 100  )
# Survived women

survived_women = train.loc[train.Sex == 'female']['Survived']

print( '% women who survived = ', (sum(survived_women)/ len(survived_women)) * 100  )
from sklearn.ensemble import RandomForestClassifier



y = train['Survived']



features = ['Sex', 'Pclass', 'SibSp', 'Parch']



X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])





model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)



predictions =  model.predict(X_test)



output = pd.DataFrame({'PassengerID': test.PassengerId, 'Survived':predictions })



output.to_csv('my_submission.csv', index=False)



print('Submission saved')