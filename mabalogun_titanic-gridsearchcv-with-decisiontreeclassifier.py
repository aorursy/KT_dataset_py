# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data = pd.get_dummies(train_data, columns = ['Sex', 'SibSp', 'Parch','Pclass','Embarked'])

train_data.head()
train_data.describe(include='all')
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
X = train_data[['Age', 'Fare', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3',

             'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2',

       'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Pclass_1', 'Pclass_2',

       'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
y = train_data.iloc[:,1]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
X_train.shape
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

dt=DecisionTreeClassifier()
parameter_grid = {'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10,15,20,30,40,50],

                  'max_features': [1, 2, 3, 4,5,6,7,8,9,10],

                 'random_state':[0, 1, 2, 3, 4, 5, 10, 15,20,35,50,80,100,150,180,200],

                 'criterion':['gini','entropy'],

                 }



grid_search = GridSearchCV(dt, param_grid = parameter_grid,

                          cv =10)



grid_search.fit(X_train, y_train)



print ("Best Score: {}".format(grid_search.best_score_))

print ("Best params: {}".format(grid_search.best_params_))
dt=DecisionTreeClassifier(max_depth=7,criterion='gini',max_features=9,random_state=0)
dt.fit(X_train,y_train)

pred=dt.predict(X_test)
pred.shape
from sklearn.metrics import classification_report,log_loss,f1_score, accuracy_score

print(classification_report(y_test,pred))

print('\n')

print('F1-SCORE : ',f1_score(y_test,pred,average=None))

print('\n')

print('Train Accuracy: ', accuracy_score(y_train, dt.predict(X_train))*100,'%')
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.describe(include='all')
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
test_data.describe(include='all')
test_data = pd.get_dummies(test_data, columns = ['Sex', 'SibSp', 'Parch','Pclass','Embarked'])

test_data.head()
test_data.shape
X_t = test_data[['Age', 'Fare', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3',

             'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2',

       'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Pclass_1', 'Pclass_2',

       'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
X_t.shape
from sklearn import preprocessing

X_t = preprocessing.StandardScaler().fit(X_t).transform(X_t)

X_t[0:5]
dt.fit(X,y)

prediction=dt.predict(X_t)
prediction.shape




output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")