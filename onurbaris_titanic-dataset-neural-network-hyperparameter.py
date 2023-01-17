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
y = train_data["Survived"]

train_data["Title"] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data["Title"] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data["familysize"]=train_data["SibSp"]+train_data["Parch"]+1
test_data["familysize"]=test_data["SibSp"]+test_data["Parch"]+1

features = ["Pclass", "Sex", "Fare", "Title", "familysize", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X_test['Title_Capt']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Countess']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Don']=X_test.apply(lambda x: 0, axis=1)
X['Title_Dona']=X.apply(lambda x: 0, axis=1)
X_test['Title_Jonkheer']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Lady']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Major']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Mlle']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Mme']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Sir']=X_test.apply(lambda x: 0, axis=1)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_scaled = X
X_scaled_test = X_test
X_scaled[['Pclass','Fare','familysize','Age']] = scaler.fit_transform(X_scaled[['Pclass','Fare','familysize','Age']])
X_scaled_test[['Pclass','Fare','familysize','Age']] = scaler.fit_transform(X_scaled_test[['Pclass','Fare','familysize','Age']])

X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())
X_scaled = X_scaled.fillna(X_scaled.median())
X_scaled_test = X_scaled_test.fillna(X_scaled_test.median())

model = MLPClassifier()
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, f1_score

param_grid = {
    'alpha' : [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    'hidden_layer_sizes' : [(20,5), (20,6), (25,6), (25,7), (30,6), (30,7)],
    'solver': ['lbfgs', 'adam', 'sgd' ],
    'max_iter': [3000]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}


def grid_search_wrapper(refit_score='f1_score'):

    """fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics"""
                      
    grid_search = GridSearchCV(model, param_grid, scoring=scorers, refit=refit_score, cv=10, return_train_score=True, n_jobs=-1)

    grid_search.fit(X_scaled, y)

    # make the predictions
    labels_pred = grid_search.predict(X_scaled)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of MLP Classifier optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y, labels_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

grid_search_clf = grid_search_wrapper()
model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(25,7), random_state=1, max_iter=3000)
model.fit(X_scaled, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

labels_pred = model.predict(X_scaled)
print(pd.DataFrame(confusion_matrix(y, labels_pred), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
