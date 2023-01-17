# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer



import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

gend_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

copy_test = test.copy()
train.head()
train.describe()
train.select_dtypes(include=['object']).describe()
test.head()
X = train.drop('Survived', axis=1)

y = train['Survived']
X.head()
#Data processing for train data

X['Sex'].replace(['male','female'],[0,1],inplace=True)

X['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

X['Family_size'] = X['SibSp'] + X['Parch']

X.drop(['Name','Ticket','Cabin','SibSp','Parch', 'PassengerId'],axis=1,inplace=True)



#Data processing for test data

test['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

test['Family_size'] = test['SibSp'] + test['Parch']

test.drop(['Name','Ticket','Cabin','SibSp','Parch', 'PassengerId'],axis=1, inplace=True)
X.head()
test.head()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')

imputed_X = pd.DataFrame(imputer.fit_transform(X))

imputed_test = pd.DataFrame(imputer.transform(test))



imputed_X.columns = X.columns

imputed_test.columns = test.columns



X = imputed_X

test = imputed_test
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_X = pd.DataFrame(scaler.fit_transform(X))

scaled_test = pd.DataFrame(scaler.transform(test))



scaled_X.columns = X.columns

scaled_test.columns = test.columns



X = scaled_X

test = scaled_test
X.head()
X.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=1)
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score



def fit_model(model):

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print(f'{accuracy_score(prediction, y_test)}' + '\n')
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier



fit_model(GaussianNB())

fit_model(LogisticRegression())

fit_model(SVC())

fit_model(DecisionTreeClassifier())

fit_model(XGBClassifier())

fit_model(RandomForestClassifier())
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation



def cross_valid_score(model):

    kfold = KFold(n_splits=10, random_state=1)

    cv_result = cross_val_score(model, X, y, cv = kfold, scoring = 'accuracy')

    print(f'the scores are: {cv_result}')

    print(f'the average score is: {sum(cv_result)/len(cv_result)} \n\n')

    

cross_valid_score(GaussianNB())

cross_valid_score(LogisticRegression())

cross_valid_score(SVC())

cross_valid_score(DecisionTreeClassifier())

cross_valid_score(XGBClassifier())

cross_valid_score(RandomForestClassifier())
from sklearn.model_selection import GridSearchCV



params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }

gd=GridSearchCV(estimator=XGBClassifier(),param_grid=params,verbose=True,cv=10)

gd.fit(X,y)

print(gd.best_score_)

print(gd.best_estimator_)
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.6, gamma=1.5, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.300000012, max_delta_step=0, max_depth=4,

              min_child_weight=5, monotone_constraints='()',

              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.6,

              tree_method='exact', validate_parameters=1, verbosity=None)



xgb.fit(X_train, y_train)

prediction = xgb.predict(test)
submission = pd.DataFrame({'PassengerId': copy_test['PassengerId'],

                          'Survived': prediction})



submission.to_csv('submission.csv', index=False)
submission.head()