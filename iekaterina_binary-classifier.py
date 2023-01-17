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

import xgboost as xgb
Titanic_pas = pd.read_csv('../input/train.csv')
#list(Titanic_pas)

Titanic_pas.head()

Titanic_pas.isnull().sum()/len(Titanic_pas)# процент отсутствующих значений в каждой колонке

# Cabin отбрасываем, т.к. 77% отсутствует

# Уберем две строки, для которых Embarked отсутствует, скорее всего две строки погоды не сделают.

Titanic_pas = Titanic_pas.loc[lambda df: ~ df['Embarked'].isnull()]

features = Titanic_pas[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#features[['Sex']] = features['Sex'] == 'male'

# добавим новую фичу - размер семьи:

features['FamilySize'] = features['SibSp'] + features['Parch'] + 1

# заполняем отсутствующие значения Age:

features.loc[features['Parch'] == 2,'Age'] = features['Age'].replace(np.NaN,features[features['Age'] <= 20]['Age'].mean()) # едет ребенок

features.loc[(features['Parch'] == 1) & (features['FamilySize'] >= 4),'Age'] = features['Age'].replace(np.NaN,features[features['Age'] < 20]['Age'].mean()) # едет ребенок

# остальные среднего возраста:

features['Age'] = features['Age'].replace(np.NaN,features[features['Age'] > 20]['Age'].mean()) # едет ребенок
Corr_matrix = features.corr()

Corr_matrix
# из показателей родственных отношений оставим только размер семьи:

features = features.drop(['Parch','SibSp'],1)
target = features['Survived']

features = features.drop(['Survived'], axis = 1)
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

features_code = features.copy()

features_code['Sex'] = lb_make.fit_transform(features['Sex'])

features_code['Embarked'] = lb_make.fit_transform(features['Embarked'])
from sklearn.model_selection import train_test_split

features_train, features_val, target_train, target_test = train_test_split(features_code, target, test_size=0.3, random_state=100)
def accuracy_score(dataset, pred):

    return (dataset == pred).sum()/len(dataset)
def depth_choosing(Xtrain, Xtest, Ytrain, Ytest, my_depth):

    model = xgb.XGBClassifier(max_depth = my_depth, n_estimators=300, learning_rate=0.005).fit(Xtrain, Ytrain)

    test_predict = model.predict(Xtest)

    return accuracy_score(Ytest, test_predict)
max_acc = depth_choosing(features_train, features_val, target_train, target_test, 1)

acc = max_acc

best_depth = 1

for depth in range(2,11):

    acc = depth_choosing(features_train, features_val, target_train, target_test, depth)

    if acc > max_acc:

        max_acc = acc

        best_depth = depth
print(max_acc)

best_model = xgb.XGBClassifier(max_depth = best_depth, n_estimators=300, learning_rate=0.05).fit(features_code, target)
Titanic_test = pd.read_csv('../input/test.csv')
Titanic_test.isnull().sum()/len(Titanic_pas)
features_test = Titanic_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

# добавим новую фичу - размер семьи:

features_test['FamilySize'] = features_test['SibSp'] + features_test['Parch'] + 1

# заполняем отсутствующие значения Age:

features_test.loc[features_test['Parch'] == 2,'Age'] = features_test['Age'].replace(np.NaN,features_test[features_test['Age'] <= 20]['Age'].mean()) # едет ребенок

features_test.loc[(features_test['Parch'] == 1) & (features_test['FamilySize'] >= 4),'Age'] = features_test['Age'].replace(np.NaN,features_test[features_test['Age'] < 20]['Age'].mean()) # едет ребенок

# остальные среднего возраста:

features_test['Age'] = features_test['Age'].replace(np.NaN,features_test[features_test['Age'] > 20]['Age'].mean()) # едет ребенок

features_test = features_test.drop(['Parch','SibSp'],1)
features_test['Fare'] = features_test['Fare'].replace(np.NaN,features_test['Fare'].mean())
features_test['Sex'] = lb_make.fit_transform(features_test['Sex'])

features_test['Embarked'] = lb_make.fit_transform(features_test['Embarked'])
test_prediction = best_model.predict(features_test)

submission = pd.DataFrame({

        "PassengerId": Titanic_test["PassengerId"],

        "Survived": test_prediction

    })

submission.to_csv('submission.csv', index=False)
submission['Survived'].sum()/len(submission)
Titanic_gender = pd.read_csv('../input/gender_submission.csv')
gender_pred = test_prediction[features_test['Sex'] == 1]

len(gender_pred)
(test_prediction == Titanic_gender['Survived']).sum()/len(test_prediction)