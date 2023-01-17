import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test_full = pd.read_csv('/kaggle/input/titanic/test.csv')

test = test_full.copy()
train.head()
print('-----Information-----')

print(train.info())
sns.distplot(train['Survived'], kde=False)
sns.heatmap(train.isna(), cmap='viridis', cbar=None, yticklabels=False)
sns.heatmap(test.isna(), cmap='viridis', cbar=None, yticklabels=False)
train['Age'].fillna(train['Age'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)



test['Fare'].fillna(test['Fare'].median(), inplace=True)
train['Cabin'].fillna(0, inplace=True)

train['Cabin'] = train['Cabin'].map(lambda p: 1 if p is not 0 else 0)



test['Cabin'].fillna(0, inplace=True)

test['Cabin'] = test['Cabin'].map(lambda p: 1 if p is not 0 else 0)
sns.heatmap(train.isna(), cmap='viridis', cbar=None, yticklabels=False)
sns.heatmap(test.isna(), cmap='viridis', cbar=None, yticklabels=False)
for i, col in enumerate(['SibSp','Parch']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=train, kind='point', aspect=2)
train['Family_count'] = train['SibSp'] + train['Parch']

test['Family_count'] = test['SibSp'] + test['Parch']

train.head()
train.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Embarked'], inplace=True)

test.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Embarked'], inplace=True)



train.head()
label = LabelEncoder()

train['Sex_conv'] = label.fit_transform(train['Sex'])

train.drop(columns='Sex', inplace=True)



test['Sex_conv'] = label.fit_transform(test['Sex'])

test.drop(columns='Sex', inplace=True)



train.head()
sns.heatmap(np.abs(train.corr()), cmap='Blues', annot=True)
X_train = train.drop(columns='Survived')

X_test = test



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif["features"] = X_train.columns

vif.round(2)
X_train.head()
scaler = StandardScaler()



X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = X_test.columns)



print('----Training Set-----')

print(X_train.head())

print('----Test Set-----')

print(X_test.head())
y = train['Survived']



X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size = 0.4, stratify=y, shuffle=True)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}



lr = GridSearchCV(LogisticRegression(), params, cv=5)

lr.fit(X_train, y_train)

lr.best_params_
LR_model = lr.best_estimator_
params = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 

          'kernel':['rbf', 'linear', 'poly', 'sigmoid']}



svc = GridSearchCV(SVC(), params, cv=5)

svc.fit(X_train, y_train)

svc.best_params_
SVC_model = svc.best_estimator_
params = {'n_estimators':[5, 50, 250, 500], 

          'max_depth':[2, 4, 8, 16, 32, None]}



rf = GridSearchCV(RandomForestClassifier(), params, cv=5)

rf.fit(X_train, y_train)

rf.best_params_
RF_model = rf.best_estimator_
params = {'learning_rate': [0.001, 0.1, 1, 10, 100], 

          'n_estimators':[5, 50, 250, 500], 

          'max_depth':[1, 3, 5, 7, 9]}



gb = GridSearchCV(GradientBoostingClassifier(), params, cv=5)

gb.fit(X_train, y_train)

gb.best_params_
GB_model = gb.best_estimator_
params = {'hidden_layer_sizes': [(10,), (50,), (100,)],

          'activation': ['relu', 'tanh', 'logistic'],

          'learning_rate': ['constant', 'invscaling', 'adaptive']}



mlp = GridSearchCV(MLPClassifier(), params, cv=5)

mlp.fit(X_train, y_train)

mlp.best_params_
MLP_model = mlp.best_estimator_
from sklearn.metrics import accuracy_score

from time import time



summary = pd.DataFrame(columns=['Model', 'Accuracy Score', 'Prediction Time (ms)'])

models = {'LR':LR_model, 'SVC':SVC_model, 'RF':RF_model, 'GB':GB_model, 'MLP':MLP_model}
def evaluate_model(name, model):

    start=time()

    y_pred = model.predict(X_val)

    stop=time()

    accuracy = accuracy_score(y_val, y_pred)

    global summary

    summary = summary.append({'Model':name, 'Accuracy Score':np.round(accuracy,4), 'Prediction Time (ms)': (stop-start)*1000}, ignore_index=True)
for name, model in models.items():

    evaluate_model(name, model)

    

summary
predictions = GB_model.predict(X_test)

output=pd.DataFrame({'PassengerId': test_full.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)