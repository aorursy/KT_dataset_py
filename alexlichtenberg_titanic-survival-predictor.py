import os

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

from sklearn.tree.export import export_text

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import matplotlib as plt

from sklearn.impute import SimpleImputer

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

print(train.head())
train.loc[train['Sex']=='male', 'Sex_bin'] = 1

train.loc[train['Sex']=='female', 'Sex_bin'] = 0

X = train[['Pclass','Sex_bin','Age','SibSp','Parch','Fare']]

X_for_cols = X.copy()

si = SimpleImputer(missing_values=np.nan, strategy='median')

si.fit(X)

X = si.transform(X)

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(max_depth = 6)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)



print("Accuracy score for imputed Decision Tree model: " + str(accuracy_score(y_test,y_pred)))
rf = RandomForestClassifier(max_depth=4, n_estimators = 2000)

rf.fit(X_train, y_train)

y_rf_pred = rf.predict(X_test)



print("Accuracy score for imputed Random Forest model: " + str(accuracy_score(y_test, y_rf_pred)))
# Create a pd.Series of features importances

importances = pd.Series(data=rf.feature_importances_,

                        index= X_for_cols.columns)



# Sort importances

importances_sorted = importances.sort_values()



# Draw a horizontal barplot of importances_sorted

importances_sorted.plot(kind='barh', color='lightgreen')
train['Title'] = train['Name'].str.split(pat = "(\.|\,)", expand=True)[2]

train['Title'] = [string.strip() for string in train['Title']]



test['Title'] = test['Name'].str.split(pat = "(\.|\,)", expand=True)[2]

test['Title'] = [string.strip() for string in test['Title']]



print(train['Title'].value_counts())

print(test['Title'].value_counts())

train.loc[train['Title'] == 'Mme', 'Title'] = 'Mlle'

train.loc[train['Title'] == 'Ms', 'Title'] = 'Miss'

train.loc[train['Title'] == 'Major', 'Title'] = 'Officer'

train.loc[train['Title'] == 'Col', 'Title'] = 'Officer'

train.loc[train['Title'] == 'Capt', 'Title'] = 'Officer'

train.loc[train['Title'] == 'the Countess', 'Title'] = 'Countess'



print(train['Title'].value_counts())



dummies = pd.get_dummies(train['Title'])

train = pd.concat([train, dummies], axis=1)
test.loc[test['Sex']=='male', 'Sex_bin'] = 1

test.loc[test['Sex']=='female', 'Sex_bin'] = 0



test.loc[test['Title'] == 'Mme', 'Title'] = 'Mlle'

test.loc[test['Title'] == 'Dona', 'Title'] = 'Mlle'

test.loc[test['Title'] == 'Ms', 'Title'] = 'Miss'

test.loc[test['Title'] == 'Major', 'Title'] = 'Officer'

test.loc[test['Title'] == 'Col', 'Title'] = 'Officer'

test.loc[test['Title'] == 'Capt', 'Title'] = 'Officer'

test.loc[test['Title'] == 'the Countess', 'Title'] = 'Countess'



print(train['Title'].value_counts())

test_dummies = pd.get_dummies(test['Title'])

test = pd.concat([test, test_dummies], axis=1)
cols = ['Pclass','Sex_bin','Age','SibSp','Parch','Fare','Officer', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Master', 'Mlle', 'Mr',

       'Miss', 'Rev', 'Sir', 'Countess']



X_title = train[cols]

X_title_for_cols = X_title.copy()

si_t = SimpleImputer(missing_values=np.nan, strategy='median')

si_t.fit(X_title)

X_title = si_t.transform(X_title)

y = train['Survived']

X_title_train, X_title_test, y_train, y_test = train_test_split(X_title, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(max_depth=6, n_estimators = 2000)

rf.fit(X_title_train, y_train)



y_rf_pred = rf.predict(X_title_test)



print(accuracy_score(y_test, y_rf_pred))
# Create a pd.Series of features importances

importances = pd.Series(data=rf.feature_importances_,

                        index= X_title_for_cols.columns)



# Sort importances

importances_sorted = importances.sort_values()



# Draw a horizontal barplot of importances_sorted

importances_sorted.plot(kind='barh', color='lightgreen')
test['Jonkheer'] = 0

test['Don'] = 0

test['Lady'] = 0

test['Countess'] =0

test['Sir']=0
X_output = test[cols]

si_output = SimpleImputer(missing_values=np.nan, strategy='median')

si_output.fit(X_output)

X_output = si_output.transform(X_output)

y_output = rf.predict(X_output)

print(y_output)
output = pd.DataFrame(columns = ['PassengerId', 'Survived'])

output['PassengerId'] = test['PassengerId']

output['Survived'] = y_output



output['PassengerId'] = pd.to_numeric(output['PassengerId'], downcast='integer')

output['Survived'] = pd.to_numeric(output['Survived'], downcast='integer')



output.to_csv('output.csv', index=False)
