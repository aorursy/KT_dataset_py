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
survied_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
survied_df = survied_df.set_index('PassengerId')
survied_df.head()
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
# train_df = train_df.join(survied_df, on='PassengerId')
train_df = train_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
train_df = train_df.dropna()
train_df['Pclass'] = train_df['Pclass'].astype('category')
train_df
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df = test_df.join(survied_df, on='PassengerId')
test_df = test_df.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
test_df = test_df.dropna()
# train_df['Pclass'] = train_df['Pclass'].astype('category')
test_df
def getFeatureData(df):
    y = train_df['Survived'].values
    X = pd.get_dummies(train_df).drop('Survived', axis=1).values
    return X, y
X_train, y_train = getFeatureData(train_df)
X_test, y_test = getFeatureData(test_df)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def KFoldScore(reg, X, y, cv=5):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    
    accuracies = []
    
    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        reg.fit(X_train, y_train)
        y_pred = np.round(reg.predict(X_test))
        
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
    return np.mean(accuracies)
    
dt = DecisionTreeClassifier(random_state=1)
nb = GaussianNB()
nn = MLPRegressor(activation='logistic', max_iter=500)
dt_cv_score = KFoldScore(dt, X_train, y_train, cv=5)
nb_cv_score = KFoldScore(nb, X_train, y_train, cv=5)
nn_cv_score = KFoldScore(nn, X_train, y_train, cv=5)

print(f'decision tree score: {dt_cv_score}, naive score: {nb_cv_score}, NN score: {nn_cv_score}')
def evaluate(reg, X, y):
    y_pred = np.round(reg.predict(X))
    
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
print('Decision Tree Score')
evaluate(dt, X_test, y_test)
print('Naive Score')
evaluate(nb, X_test, y_test)
print('Nueral Network Score')
evaluate(nn, X_test, y_test)
