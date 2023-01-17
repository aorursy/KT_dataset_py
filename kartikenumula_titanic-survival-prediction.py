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
import math

# Data Manipulation
import pandas as pd
import numpy as np

# Visualization
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mns

# Machine learning models & utilities
from sklearn.model_selection import train_test_split as tts
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# We will ignore warnings
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
sns.set_palette('rainbow')
train = pd.read_csv(r'/kaggle/input/titanic/train.csv')
train.head()
train.shape
train.isnull().sum()
mns.matrix(train, figsize=(10,5))
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
def impute_age(cols):
    
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return 37
        if pclass == 2:
            return 29
        if pclass == 3:
            return 25
    else:
        return age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)
train.isnull().sum()
train['Cabin'].value_counts()
train['Cabin'].fillna(0, inplace = True)
train['Cabin'].head()
cabins = []
for cabin in train['Cabin']:
    cabins.append(str(cabin))
chars = []
for i in cabins:
    char = i[0]
    chars.append(char)
train['Cabin'] = chars
train['Cabin'].head()
train['Cabin'].value_counts()
train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace = True)
train['Embarked'].value_counts()
train.isnull().sum()
corr_matrix = train.corr()

fix, ax = plt.subplots(figsize=(10,8))

sns.heatmap(corr_matrix, annot = True, fmt = '.2g', vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm')
numerical_columns = ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Survived']
train[numerical_columns].hist(bins = 20, figsize = (12, 10))
plt.show()
sns.countplot(x = 'Sex', data = train)
sns.countplot(x = 'Survived', data = train)
sns.countplot(x = 'Embarked', data = train)
# S = Southampton, C = Cherbourg, Q = Queenstown
sns.countplot(x = 'Pclass', data = train)
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
sns.countplot(x = 'Survived', hue = 'Embarked', data = train)
train['Family'] = train.apply(lambda x : x['SibSp'] + x['Parch'], axis = 1)
train['Family'].value_counts()
sns.countplot(x = 'Family', hue = 'Survived', data = train)
train_df = train
train_df.drop(['SibSp', 'Parch', 'Name', 'Ticket'], axis = 1, inplace = True)
train_df.head()
train_df = pd.get_dummies(train_df)
train_df.head()
train_df.drop(['PassengerId', 'Cabin_T'], axis = 1, inplace = True)
train_df.head()
y = train_df['Survived'] # This is the value we are trying to predict
X_train = train_df.drop('Survived', axis = 1) # Training data with all columns except 'Survived'
X_train.head()
X_train.shape
def fit_ml_alg(alg, xtrain, ytrain, cv):
    
    # single pass - model training and regular accuracy score
    model = alg.fit(xtrain, ytrain)
    r_acc = round(model.score(xtrain, ytrain) * 100, 2)
    
    # cross validation
    train_preds = model_selection.cross_val_predict(alg, xtrain, ytrain, cv=cv, n_jobs=-1)
    
    # cross validation accuracy metric
    cv_acc = round(metrics.accuracy_score(ytrain, train_preds) * 100, 2)
    
    print('Accuracy: %s' % r_acc)
    print('CV 10-fold Accuracy: %s' % cv_acc)
    
    return train_preds, r_acc, cv_acc
log_model = LogisticRegression(max_iter=900)
log_preds, log_acc, log_cv_acc = fit_ml_alg(log_model, X_train, y, 10)
knn_model = KNeighborsClassifier()
knn_preds, knn_acc, knn_cv_acc = fit_ml_alg(knn_model, X_train, y, 10)
gbc_model = GradientBoostingClassifier()
gbc_preds, gbc_acc, gbc_cv_acc = fit_ml_alg(gbc_model, X_train, y, 10)
gnb_model = GaussianNB()
gnb_preds, gnb_acc, gnb_cv_acc = fit_ml_alg(gnb_model, X_train, y, 10)
svc_model = LinearSVC()
svc_preds, svc_acc, svc_cv_acc = fit_ml_alg(svc_model, X_train, y, 10)
rfc_model = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state = 7,
                                  criterion = 'entropy', class_weight = 'balanced', oob_score = True)
rfc_preds, rfc_acc, rfc_cv_acc = fit_ml_alg(rfc_model, X_train, y, 10)
sgd_model = SGDClassifier()
sgd_preds, sgd_acc, sgd_cv_acc = fit_ml_alg(sgd_model, X_train, y, 10)
dtc_model = DecisionTreeClassifier()
dtc_preds, dtc_acc, dtc_cv_acc = fit_ml_alg(dtc_model, X_train, y, 10)
models = pd.DataFrame({
    
    'Model': [
        'Logistic Regression',
        'K-Nearest Neighbors',
        'Gradient Boost',
        'Gaussian Naive Bayes',
        'Linear SVC',
        'Random Forest Classifier',
        'Stochastic Gradient Descent',
        'Decision Tree Classifier'
    ],
    
    'Score': [
        log_acc,
        knn_acc,
        gbc_acc,
        gnb_acc,
        svc_acc,
        rfc_acc,
        sgd_acc,
        dtc_acc
    ],
    
    'Cross-Val Score': [
        log_cv_acc,
        knn_cv_acc,
        gbc_cv_acc,
        gnb_cv_acc,
        svc_cv_acc,
        rfc_cv_acc,
        sgd_cv_acc,
        dtc_cv_acc
    ]
})
print('---Model Evaluation Metrics---')
models.sort_values(by='Cross-Val Score', ascending=False)
def feature_imp(model, data):
    
    f_imp = pd.DataFrame({'Importance': model.feature_importances_, 'Column': data.columns})
    
    f_imp = f_imp.sort_values(['Importance', 'Column'], ascending = [True, False]).iloc[-30:]
    
    f_imp.plot(kind = 'barh', x = 'Column', y = 'Importance', figsize = (12,10))
feature_imp(rfc_model, X_train)
test = pd.read_csv(r'/kaggle/input/titanic/test.csv')
test.head()
test.isnull().sum()
test['Age'] = test[['Age', 'Pclass']].apply(impute_age, axis = 1)
test.isnull().sum()
test['Fare'].fillna(test['Fare'].median(), inplace = True)
test.isnull().sum()
test['Cabin'].fillna(0, inplace = True)
cabins = []
for cabin in test['Cabin']:
    cabins.append(str(cabin))
chars = []
for i in cabins:
    char = i[0]
    chars.append(char)
test['Cabin'] = chars
test.isnull().sum()
test['Cabin'].value_counts()
test.head()
test['Family'] = test.apply(lambda x : x['SibSp'] + x['Parch'], axis = 1)
test.head()
test['Family'].value_counts()
test_df = test
test_df.head()
test_df.drop(['SibSp', 'Parch', 'Name', 'Ticket'], axis = 1, inplace = True)
test_df.head()
test_df = pd.get_dummies(test_df)
test_df.head()
test_columns = X_train.columns
test_columns
test_df[test_columns].head()
X_train.shape
test_df[test_columns].shape
final_preds = rfc_model.predict(test_df[test_columns])
final_preds
submission = pd.DataFrame()
submission['PassengerId'] = test_df['PassengerId']
submission['Survived'] = final_preds
submission.head()
if len(submission) == len(test_df):
    print('Submission and Test Data are of the same length!')
else:
    Print('Something wrong with the Submission length')
submission.to_csv('rfc_gender_submission.csv', index = False)
