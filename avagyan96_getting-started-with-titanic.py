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
        
import seaborn as sns

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


TempForDraw = X.copy()
TempForDraw['target'] = y
sns.pairplot(TempForDraw, hue='target')
from sklearn import linear_model, ensemble, model_selection, metrics, manifold
from sklearn import linear_model, ensemble, model_selection, metrics, manifold
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
import xgboost as xgb

rf_classifier = RandomForestClassifier(n_jobs=-1)
grid_parameters_rf = {
    'max_depth' : np.arange(95, 105),
    'n_estimators' : np.arange(55, 75),
}
cv = model_selection.StratifiedShuffleSplit(n_splits=3, test_size = 0.3, random_state = 0)
gridCV_rf = model_selection.GridSearchCV(rf_classifier, grid_parameters_rf, scoring = 'roc_auc', cv=cv)
%time gridCV_rf.fit(X, y)
print('ROC-AUC:', gridCV_rf.best_score_)
print(gridCV_rf.best_params_)

model = RandomForestClassifier(n_estimators=95, max_depth=66, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")