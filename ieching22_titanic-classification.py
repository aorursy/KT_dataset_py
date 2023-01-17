## Loading Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score
from statistics import mean
## Loading data
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
## Data Processing
def process_df(df):
    # Creating dummy variables
    class_dummy = pd.get_dummies(df['Pclass'], drop_first=True)
    sex_dummy = pd.get_dummies(df['Sex'], drop_first=True)
    embarked_dummy = pd.get_dummies(df['Embarked'], drop_first=True)

    df = df.drop(columns=['PassengerId','Name','Cabin','Ticket'])
    df = df.drop(columns=['Pclass','Sex','Embarked'])

    df = pd.concat([df, class_dummy], axis=1)
    df = pd.concat([df, sex_dummy], axis=1)
    df = pd.concat([df, embarked_dummy], axis=1)
    
    # Imputation
    col_names = df.columns
    
    imp = KNNImputer(n_neighbors=29)
    imputed = imp.fit_transform(df)
    df = pd.DataFrame(imputed, columns=col_names)
    
    # Scaling
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df = pd.DataFrame(scaled, columns=col_names)
    
    return df
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

X = process_df(X)
test_id = test_df['PassengerId']
test_df = process_df(test_df)
#X_train, X_test, y_train, y_test = train_test_split(X, y)
## Random Forest Model
score_list = []
roc_list = []
kf = KFold(5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    score_list.append(clf.score(X_test, y_test))
    y_score = clf.predict(X_test)
    roc_list.append(roc_auc_score(y_test, y_score))
print('Mean R squared: ', mean(score_list))
print('Mean ROC: ', mean(roc_list))
## Gradient Boosting Model
score_list = []
roc_list = []
kf = KFold(5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    score_list.append(clf.score(X_test, y_test))
    y_score = clf.predict(X_test)
    roc_list.append(roc_auc_score(y_test, y_score))
print('Mean R squared: ', mean(score_list))
print('Mean ROC: ', mean(roc_list))
## Using Random Search for Hyperparameter Tuning
forest_grid = {'n_estimators': [100, 200, 300],
               'max_features': ['auto', 'sqrt', 'log2'],
               'min_samples_split': [2, 5, 10],
               'bootstrap': [True, False]}

clf = RandomizedSearchCV(RandomForestClassifier(),
                         param_distributions = forest_grid,
                         cv = 5,
                         n_jobs = -1)

score_list = []
roc_list = []
kf = KFold(5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf.fit(X_train, y_train)
    score_list.append(clf.score(X_test, y_test))
    y_score = clf.predict(X_test)
    roc_list.append(roc_auc_score(y_test, y_score))
    
    print('Finished Fold')
print('Mean R squared: ', mean(score_list))
print('Mean ROC: ', mean(roc_list))
boosting_grid = {'n_estimators': [100, 200, 300],
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'min_samples_split': [2, 5, 10],
                 'learning_rate': [0.1, 0.01]}

clf = RandomizedSearchCV(GradientBoostingClassifier(),
                         param_distributions = boosting_grid,
                         cv = 5,
                         n_jobs = -1)

score_list = []
roc_list = []
kf = KFold(5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf.fit(X_train, y_train)
    score_list.append(clf.score(X_test, y_test))
    y_score = clf.predict(X_test)
    roc_list.append(roc_auc_score(y_test, y_score))
    
    print('Finished Fold')
print('Mean R squared: ', mean(score_list))
print('Mean ROC: ', mean(roc_list))
## Making Predictions
boosting_grid = {'n_estimators': [100, 200, 300],
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'min_samples_split': [2, 5, 10],
                 'learning_rate': [0.1, 0.01]}

clf = RandomizedSearchCV(GradientBoostingClassifier(),
                         param_distributions = boosting_grid,
                         cv = 5,
                         n_jobs = -1)

clf.fit(X, y)
y_pred = clf.predict(test_df)

final = pd.DataFrame({'PassengerId': test_id, 
                      'Survived': y_pred})
final.head()
final.to_csv("submission.csv", index=False)
