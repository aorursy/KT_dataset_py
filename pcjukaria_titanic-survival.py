import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

%config Completer.use_jedi = False
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.info()
train.drop(['PassengerId', 'Cabin'], axis = 1, inplace = True)

test.drop(['PassengerId', 'Cabin'], axis = 1, inplace = True)
train.drop(['Ticket'], axis = 1, inplace = True)

test.drop(['Ticket'], axis = 1, inplace = True)
train.head()
train.drop(['Name'], axis = 1, inplace = True)

test.drop(['Name'], axis = 1, inplace = True)
train.head()
train.info()
train.describe()

train['alone'] = train.Parch + train.SibSp

def CreateCat(s):

    if s>0 : return 1

    else : return 0

    

train['alone'] = train['alone'].apply(CreateCat)

test['alone'] = test.Parch + test.SibSp

def CreateCat(s):

    if s>0 : return 1

    else : return 0

    

test['alone'] = test['alone'].apply(CreateCat)
train.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

test.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
train.head()
train.info()
train['Embarked'] = train['Embarked'].fillna('S')
from sklearn.model_selection import train_test_split
X, y = train.drop('Survived', axis=1), train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
num_cols = ['Age','Fare']

cat_cols = ['Sex','Embarked']



num_pipe = Pipeline([('impute', SimpleImputer(strategy='mean')),

                     ('scale', StandardScaler())])



cat_pipe = Pipeline([('encode', OneHotEncoder())])



full_pipe = ColumnTransformer([('num', num_pipe, num_cols),

                               ('cat', cat_pipe, cat_cols)])





X_train = pd.concat([pd.DataFrame(full_pipe.fit_transform(X_train), columns=['Age','Fare', 'female', 'male', 'C', 'R','S' ]),

           pd.DataFrame(X_train['Pclass'].values, columns=['Pclass']),

          pd.DataFrame(X_train['alone'].values, columns=['alone'])], axis = 1)
X_test = pd.concat([pd.DataFrame(full_pipe.fit_transform(X_test), columns=['Age','Fare', 'female', 'male', 'C', 'R','S' ]),

           pd.DataFrame(X_test['Pclass'].values, columns=['Pclass']),

          pd.DataFrame(X_test['alone'].values, columns=['alone'])], axis = 1)
test = pd.concat([pd.DataFrame(full_pipe.fit_transform(test), columns=['Age','Fare', 'female', 'male', 'C', 'R','S' ]),

           pd.DataFrame(test['Pclass'].values, columns=['Pclass']),

          pd.DataFrame(test['alone'].values, columns=['alone'])], axis = 1)
from sklearn.ensemble import RandomForestClassifier
forest  = RandomForestClassifier(n_estimators=100, max_depth=30, oob_score=True, verbose=100 )
forest.fit(X_train, y_train)
forest.oob_score_
params = [{'n_estimators': [50, 100, 200], 'max_depth' : [20, 40, 60]},

          {'bootstrap': [False], 'n_estimators': [40, 80, 150], 'max_depth' : [30, 50, 70]} ]



from sklearn.model_selection import GridSearchCV
gsearch = GridSearchCV(RandomForestClassifier(), params,  cv = 5, verbose = 10)
gsearch.fit(X_train, y_train)
gsearch.best_estimator_
gsearch.best_params_
gsearch.best_score_
from sklearn.svm import SVC
svc_params = [{'C': [0.5, 1, 10, 100],'degree': [3,5,10,20], 'gamma': [0.1,0.5,0.01, 0.8, 1] }, 

              {'kernel': ['poly'],'C': [0.5, 1, 10],'degree': [3,5,10], 'gamma':['auto']}]



gsearch_svc = GridSearchCV(SVC(), svc_params, scoring='accuracy', cv=5, verbose=10)
gsearch_svc.fit(X_train, y_train)
gsearch_svc.best_estimator_
gsearch_svc.best_params_
gsearch_svc.best_score_
final_model = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',

    max_iter=-1, probability=True, random_state=None, shrinking=True,

    tol=0.001, verbose=False)
final_model.fit(X_train, y_train)
from sklearn.model_selection import cross_val_predict
y_proba = cross_val_predict(final_model, X_train, y_train, cv = 10, verbose = 10, method = 'predict_proba')
from sklearn.metrics import precision_recall_curve
p, r, t = precision_recall_curve(y_train, y_proba[:, 1])
plt.figure(figsize=(12,8))

plt.plot(t, p[:-1], label = 'precision')

plt.plot(t, r[:-1], label = 'recall')

plt.legend()
plt.plot(p,r)

plt.xlabel('precision')

plt.ylabel('recall')
y_proba_test = final_model.predict_proba(X_test)

y_predict = (y_proba_test[:,1]>0.85).astype('int')
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_predict))

print('\n')

print(metrics.classification_report(y_test, y_predict))
test
y_test_proba = final_model.predict_proba(test)

y_test_predict = (y_test_proba[:,1]>=0.85).astype('int')

y_test_predict
test1 = pd.read_csv("../input/titanic/test.csv")
my_submission_titanic = pd.DataFrame({'PassengerId': test1.PassengerId, 'Survived': y_test_predict})

# you could use any filename. We choose submission here

my_submission_titanic.to_csv('submission.csv', index=False)