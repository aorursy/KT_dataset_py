# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree, ensemble, preprocessing, model_selection, feature_selection

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.chdir("../input")
# Any results you write to the current directory are saved as output.
train = pd.read_csv(r'train.csv')
test = pd.read_csv(r'test.csv')
train.Age = preprocessing.Imputer().fit_transform(train[['Age']])
test.Age = preprocessing.Imputer().fit_transform(test[['Age']])
test.Fare = preprocessing.Imputer().fit_transform(test[['Fare']])
combo = pd.concat([train,test], axis=0, sort=False)
combo.loc[combo['Embarked'].isna(), 'Embarked'] = combo.Embarked.mode()
combo['FamilyCount'] = combo['SibSp'] + combo['Parch'] + 1
combo['FamilySize'] = combo['FamilyCount'].apply(lambda val : 'Single' if val < 2 else 'Small' if val < 4 else 'large' if val < 5 else 'Big')
combo['AgeType'] = combo['Age'].apply(lambda val : 'Child' if val < 12 else 'Teen' if val <= 21 else 'Young' if val < 30 else 'Middle' if val < 45 else 'Senior citizen' if val < 60 else 'Old')
combo['Title'] = combo['Name'].apply(lambda name : name.split(',')[1].split('.')[0])
combo = pd.get_dummies(data = combo, columns = ['Embarked', 'Pclass','Title','FamilySize','AgeType','Sex'])
train = combo[:891]
test = combo[891:]
test.drop(columns=['Survived'], axis=1, inplace=True)
X_train = train.drop(columns=['Name','PassengerId','SibSp','Parch','Age','Fare','Cabin', 'Ticket', 'Survived'], axis=1)
y_train = train['Survived']
X_test = test.drop(columns=['Name','PassengerId','SibSp','Parch','Fare','Age','Cabin', 'Ticket'], axis=1)
classifer = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[3,4,5,6], 'criterion':['gini','entropy']}
grid_classifier = model_selection.GridSearchCV(classifer, param_grid=dt_grid, cv=10, refit=True, return_train_score=True)
grid_classifier.fit(X_train, y_train)
results = grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score').mean())
print(results.get('mean_train_score').mean())
final_model = grid_classifier.best_estimator_
test['Survived'] = final_model.predict(X_test)
submission = pd.DataFrame(data = test[['PassengerId', 'Survived']],columns = ['PassengerId', 'Survived'])