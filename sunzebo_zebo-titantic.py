import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn import feature_selection
from sklearn import metrics
from sklearn import linear_model, ensemble, gaussian_process
from xgboost import XGBClassifier
df_train_orig = pd.read_csv('../input/train.csv')
df_test_orig = pd.read_csv('../input/test.csv')

df_train = df_train_orig.copy(deep=True)
df_train.name = 'Training set'
df_test = df_test_orig.copy(deep=True)
df_test.name = 'Test set'
print(df_train_orig.info())
df_train_orig.sample(10)
def show_nulls(df):
    print('{} columns with null values '.format(df.name))
    print(df.isnull().sum())
    print("\n")
    
for df in [df_train, df_test]:
    show_nulls(df)
for df in [df_train, df_test]:    
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
df_train.drop(['PassengerId','Cabin', 'Ticket'], axis=1, inplace=True)
for df in [df_train, df_test]:
    show_nulls(df)
df_survive = df_train_orig['Survived'].value_counts()
print(df_survive)
ax = df_survive.plot.bar()
ax.set_xticklabels(('Not Survived', 'Survived'))
for df in [df_train, df_test]:    
    df['Family_Members'] = df['SibSp'] + df['Parch'] + 1
    
    df['Is_Alone'] = 1
    df['Is_Alone'].loc[df['Family_Members'] > 1] = 0
    
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0] 

df_train.sample(10)
df_train['Title'].value_counts()
train_title_names = (df_train['Title'].value_counts() < 10)
df_train['Title'] = df_train['Title'].apply(lambda x: 'Other' if train_title_names.loc[x] == True else x)

df_train['Title'].value_counts()
test_title_names = (df_test['Title'].value_counts() < 10)
df_test['Title'] = df_test['Title'].apply(lambda x: 'Other' if test_title_names.loc[x] == True else x)

df_test['Title'].value_counts()
le = LabelEncoder()
for df in [df_train, df_test]:    
    df['Sex_Label'] = le.fit_transform(df['Sex'])
    df['Embarked_Label'] = le.fit_transform(df['Embarked'])
    df['Title_Label'] = le.fit_transform(df['Title'])
    
df_train.head()
X_cols = ['Sex', 'Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'Family_Members', 'Is_Alone']

df_train_dummy = pd.get_dummies(df_train[X_cols])
df_test_dummy = pd.get_dummies(df_test[X_cols])
df_train_dummy['Survived'] = df_train['Survived']

df_train_dummy.head()
X_train = df_train_dummy.drop(['Survived'], axis=1)
Y_train = df_train_dummy['Survived']
seed = 0

models = [ensemble.RandomForestClassifier(n_estimators=65, min_impurity_decrease=0.1, random_state=seed),
          ensemble.AdaBoostClassifier(n_estimators=11, algorithm='SAMME.R'),
          ensemble.GradientBoostingClassifier(loss='exponential', learning_rate=0.01, n_estimators=100, criterion='friedman_mse', max_depth=4),
         linear_model.LogisticRegressionCV(cv=3, penalty='l2', solver='newton-cg'),
         XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)]

fits = [model.fit(X_train, Y_train) for model in models]

fits
Y_hats = {model.__class__.__name__: model.predict(X_train) for model in models}
cv_split = ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=seed)
cv_split
for model in models:
    cv_results = cross_validate(model, X_train, Y_train, cv=cv_split)
    print(model.__class__.__name__)
    print(cv_results)
for model, Y_hat in Y_hats.items():
    print(model)
    print(metrics.classification_report(Y_train, Y_hat, target_names=['Not Survived', 'Survived']))
    print('\n')
submission_model = models[4]
submission_model
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = df_test_orig['PassengerId']
submission_df['Survived'] = submission_model.predict(df_test_dummy)
submission_df.head(10)
submission_df.to_csv('submissions.csv', header=True, index=False)