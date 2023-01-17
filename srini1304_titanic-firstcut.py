# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.columns
train_df.info()
class_sex_df = pd.DataFrame(train_df.groupby(['Pclass', 'Sex'])['Age'].aggregate('mean')).reset_index()
class_sex_df
train_df.groupby('Pclass')['Age'].aggregate('mean')
def fill_age(X):

  if(X == 1.0):

    return 38.23

  elif(X == 2.0):

    return 29.87

  else:

    return 25.14
import seaborn as sns
sns.heatmap(train_df.isnull(), yticklabels='False', cmap='coolwarm')
train_df['Age'].fillna(train_df['Pclass'].apply(fill_age), inplace=True)
sns.heatmap(train_df.isnull(), yticklabels=False)
train_df.drop('Cabin', axis=1, inplace=True)
sns.heatmap(train_df.isnull(), yticklabels=False)
train_df.dropna(inplace=True)
sns.heatmap(train_df.isnull(), yticklabels=True)
sex = pd.get_dummies(train_df['Sex'], drop_first=True)
embark =pd.get_dummies(train_df['Embarked'], drop_first=True)
train_df = pd.concat([train_df, embark, sex],axis=1)
from sklearn.cross_validation import train_test_split
name_col = train_df['Name']
ticket_col = train_df['Ticket']
train_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)
train_df.drop('Name', axis=1, inplace=True)
train_df.drop('Ticket', axis=1, inplace=True)
train_df.columns
train_df.info()
X = train_df.drop('Survived', axis=1)
Y = train_df['Survived']
X.head()
X.describe()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
preds = log_reg.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(Y_test, preds)
print(classification_report(Y_test, preds))
name_col.head(10)
def return_salutation(X):

    X_arr = X.split()

    if('Mrs.' in X_arr):

        return 'Mrs.'

    elif('Mr.' in X_arr):

        return 'Mr.'

    elif('Master.' in X_arr):

        return 'Master.'

    elif('Miss.' in X_arr):

        return 'Miss.'

    elif('Ms.' in X_arr):

        return 'Miss.'

    elif('Dr.' in X_arr):

        return 'Dr.'

    elif('Rev.' in X_arr):

        return 'Rev.'

    else:

        return 'none'
name_temp = name_col[10]
name_temp.split()
return_salutation(name_temp)
salutation_col = name_col.apply(return_salutation)
type(salutation_col)
salutation_col.value_counts()
name_df = pd.DataFrame(name_col)
name_df.head()
name_df['salutation'] = salutation_col
name_df.head()
name_df[name_df['salutation'] == 'none']['Name']
title = pd.get_dummies(name_df['salutation'], drop_first=True)
train_df = pd.concat([train_df, title], axis=1)
train_df.head()
X = train_df.drop(['Survived', 'PassengerId'], axis=1)
Y = train_df['Survived']
log_reg_1 = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=101)
log_reg_1.fit(X_train, Y_train)
preds_1 = log_reg_1.predict(X_test)
print(confusion_matrix(Y_test, preds_1))
print(classification_report(Y_test, preds_1))
test_df = pd.read_csv('../input/test.csv')
embark_test = pd.get_dummies(test_df['Embarked'], drop_first=True)
sex_test = pd.get_dummies(test_df['Sex'], drop_first=True)
test_df = pd.concat([test_df, embark_test, sex_test], axis=1)
test_df.head()
name_test_col = test_df['Name']
salutation_test_col = name_test_col.apply(return_salutation)
salutation_mod_cols = pd.get_dummies(salutation_test_col, drop_first=True)
test_df = pd.concat([test_df, salutation_mod_cols], axis=1)
test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.head()
X.head()
test_df.drop(['Embarked', 'Sex'], axis=1, inplace=True)
test_df.head()
test_df['Age'].fillna(test_df['Pclass'].apply(fill_age), inplace=True)
test_df.groupby('Pclass')['Fare'].aggregate('mean')
test_df[test_df['Fare'].isnull()]
test_df['Fare'].fillna(12.46, inplace=True)
test_df.head()
preds_test = log_reg_1.predict(test_df)
test_df_reread = pd.read_csv('../input/test.csv')
test_df_reread.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_df_reread.head()
test_df_reread['Survived'] = preds_test
test_df_reread.head()
test_df_reread.to_csv("submission.csv", index=False)
train_df['Age'].describe()
def return_age_quartile(X):

    if(X < 22):

        return 1

    elif(X < 26):

        return 2

    elif(X < 36.5):

        return 3

    else:

        return 4
train_df['age_quartile'] = train_df['Age'].apply(return_age_quartile)
train_df['Fare'].describe()
def return_fare_quartile(X):

    if(X < 7.89):

        return 1

    elif(X < 14.45):

        return 2

    elif(X < 31):

        return 3

    else:

        return 4
train_df['fare_quartile'] = train_df['Fare'].apply(return_fare_quartile)
train_df.columns
X = train_df.drop(['PassengerId', 'Survived', 'Age', 'Fare'], axis =1)
Y = train_df['Survived']
def return_length(X):

    return len(X)
X['name_len'] = name_col.apply(return_length)
X['name_len'].describe()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=101)
log_pred_2 = LogisticRegression()
log_pred_2.fit(X_train, Y_train)
y_preds_l = log_pred_2.predict(X_test)
print(confusion_matrix(Y_test, y_preds_l))
print(classification_report(Y_test, y_preds_l))
from sklearn.svm import SVC
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, Y_train)
y_preds_svl = svc.predict(X_test)
print(confusion_matrix(Y_test, y_preds_svl))
print(classification_report(Y_test, y_preds_svl))
np.sum(y_preds_l != y_preds_svl)
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_preds_knn5 = knn.predict(X_test)
print(confusion_matrix(Y_test, y_preds_knn5))
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
svm_clf = Pipeline((("scaler", StandardScaler()), ("linear_svc", SVC(kernel="linear"))))
svm_clf.fit(X_train, Y_train)
y_preds_svl_scaled = svm_clf.predict(X_test)
print(confusion_matrix(Y_test, y_preds_svl_scaled))
svm_rbf_clf = Pipeline((("scaler", StandardScaler()), ("linear_svc", SVC(kernel="rbf", probability=True))))
svm_rbf_clf.fit(X_train, Y_train)
y_preds_sv_rbf_scaled = svm_rbf_clf.predict(X_test)
print(confusion_matrix(Y_test, y_preds_sv_rbf_scaled))
from sklearn.model_selection import GridSearchCV
param_grid =[{'C':[0.1, 1, 2, 5, 10], 'kernel':['linear']}, {'kernel': ['rbf'], 'gamma':[0.001, 0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2], 'C':[1, 3, 5, 8, 10, 12, 14, 20]}]
svc_cv_clf = SVC()
grid_search = GridSearchCV(svc_cv_clf, param_grid, cv=5, scoring="average_precision")
grid_search.fit(X_train, Y_train)
grid_search.best_estimator_
y_preds_sv_rbf_cv = grid_search.best_estimator_.predict(X_test)
print(confusion_matrix(Y_test, y_preds_sv_rbf_cv))
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, Y_train)
y_preds_rnd_clf = rnd_clf.predict(X_test)
print(confusion_matrix(Y_test, y_preds_rnd_clf))
from sklearn.ensemble import VotingClassifier
vote_clf = VotingClassifier(estimators=[('lr', log_pred_2), ('sv_rbf_clf', svm_rbf_clf), ('sv_l_clf', svc)], voting='hard')
vote_clf.fit(X_train, Y_train)
y_preds_vote = vote_clf.predict(X_test)
print(confusion_matrix(Y_test, y_preds_vote))
from sklearn.tree import DecisionTreeClassifier
dec_clf = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=16)
dec_clf.fit(X_train, Y_train)
dec_clf.get_params
y_preds_dec_clf = dec_clf.predict(X_test)
print(confusion_matrix(Y_test, y_preds_dec_clf))
vote_clf = VotingClassifier(estimators=[('lr', log_pred_2), ('sv_rbf_clf', svm_rbf_clf), ('sv_l_clf', svc), ('dec_clf', dec_clf)], voting='soft')
vote_clf.fit(X_train, Y_train)
y_preds_vote_1 = vote_clf.predict(X_test)
print(confusion_matrix(Y_test, y_preds_vote_1))
temp_df = X_test.copy()
temp_df['y_test'] = Y_test
temp_df['y_preds'] = y_preds_vote_1
X_test.head()
X