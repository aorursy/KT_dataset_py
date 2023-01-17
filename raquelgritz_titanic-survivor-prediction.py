# Titanic Kaggle competition Notebook
!pip install shap
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import shap



shap.initjs()



from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def receive_variables_transformed(x, columns, df, categorical):

    while x < len(columns):

        df[columns[x]] = categorical[columns[x]]

        

        x += 1



    return df
def scale_data(df):

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df)

    df.loc[:,:] = scaled_values



    return df
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

example_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head(2)
train.shape
train.describe()
test.head(2)
test.shape
example_submission.head(2)

del example_submission
sns.set_style('darkgrid')
train['Age'].hist(bins=50)
sns.countplot(x='Survived', data=train)
sns.countplot(x='Pclass', data = train)
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train, kind="bar",

            ci="sd", palette="dark", alpha=.6, height=6)
train.head()
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

train.head(2)
test.head()
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.head(2)
null_values = train.isnull().sum()

print(null_values[null_values > 0])

del null_values
train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))

train.head()
train.isnull().sum()
null_values = test.isnull().sum()

print(null_values[null_values > 0])

del null_values
test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))

test.head()
test.isnull().sum()
train.dtypes
categorical = train.select_dtypes(include = [object])

categorical.head()
categorical = categorical.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()].astype(str)), index=series[series.notnull()].index))

categorical.head()
categorical_columns = categorical.columns

receive_variables_transformed(0, categorical_columns, train, categorical)
del categorical, categorical_columns
test.dtypes
categorical = test.select_dtypes(include = [object])

categorical.head()
categorical = categorical.apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()].astype(str)), index=series[series.notnull()].index))

categorical.head()
categorical_columns = categorical.columns

receive_variables_transformed(0, categorical_columns, test, categorical)
del categorical, categorical_columns, receive_variables_transformed
features_train = train[['Survived', 'PassengerId']]

train.drop(['Survived', 'PassengerId'], axis=1, inplace=True)

print(train.head(2))
features_test = test['PassengerId']

test.drop(['PassengerId'], axis=1, inplace=True)

print(test.head(2))
train = scale_data(train)

test = scale_data(test)
df_train = pd.concat([train, features_train], axis=1)

del train



train = df_train.copy()

del df_train, features_train
train.head(2)
df_test = pd.concat([test, features_test], axis=1)

del test



test = df_test.copy()

del df_test
test.head(2)
X = train.drop(['Survived'], axis=1)

y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
tree = DecisionTreeClassifier()
# params = {

#     'criterion': ['gini', 'entropy'],

#     'max_depth': [2, 3, 4, 5],

#     'max_features': ['sqrt', 'log2', None]

# }



# grid_tree = GridSearchCV(estimator=tree,

#                         param_grid=params,

#                         scoring='f1',

#                         cv=10,

#                         verbose=1,

#                         n_jobs=-1)
#grid_tree.fit(X_train, y_train)

tree.fit(X_train, y_train)
#print(grid_tree.best_estimator_)
#y_pred = grid_tree.predict(X_test)

y_pred = tree.predict(X_test)
print(pd.crosstab(y_test, y_pred, rownames = ['Real Value'], colnames = ['Predicted Value'], margins = True), '\n\n')
print(classification_report(y_test, y_pred))
#grid_tree.score(X_test, y_test)

tree.score(X_test, y_test)
explainer = shap.TreeExplainer(tree)

shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value[0], shap_values[0], X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
survivor_prediction = pd.DataFrame()

survivor_prediction['PassengerId'] = test['PassengerId']

survivor_prediction['Survived'] = tree.predict(test)
survivor_prediction.head()
survivor_prediction['Survived'].value_counts()
survivor_prediction.shape
survivor_prediction.to_csv('submission.csv', index=False)