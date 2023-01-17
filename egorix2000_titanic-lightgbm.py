import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.metrics import accuracy_score

RANDOM_STATE = 17
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_all = pd.concat([train, test], sort=True).reset_index(drop=True)
df_all.head()
def divide_df(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
print(train.info())
print(test.info())
def display_missing(data, name):
    missing_data = data.isna().sum()
    print(name)
    for feature, i in missing_data.items():
        print('%s column missing values: %s' % (feature, i))
    print('\n')



display_missing(train, 'Train')
display_missing(test, 'Test')
numerical = list(set(df_all.columns) - 
                 set(['Sex', 'Embarked', 'Survived']))

corr_matrix = df_all[numerical].corr()
fig, ax = plt.subplots(1,3,figsize=(15,4))
sns.heatmap(corr_matrix, annot=True,  ax=ax[0], fmt=".2f");
sns.boxplot(x='Sex', y='Age', data=df_all, ax=ax[1]);
sns.boxplot(x='Embarked', y='Age', data=df_all, ax=ax[2]);
fig.show()
df_all['Age'] = df_all.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
df_all['Fare'] = df_all.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.median()))
embarked_mode = df_all['Embarked'].mode()[0]
df_all['Embarked'] = df_all['Embarked'].fillna(embarked_mode)
print('Missing values of Embarked filled with:', embarked_mode)
df_all.drop('Cabin', axis=1, inplace=True)
fig1, ax1 = plt.subplots()
ax1.pie(train['Survived'].groupby(train['Survived']).count(), 
        labels = ['Not Survived', 'Survived'], autopct = '%1.1f%%')
ax1.axis('equal')

plt.show()
df_all.head()
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
fig, ax = plt.subplots(figsize=(15,5))
sns.countplot(x='Title', hue='Survived', data=df_all, ax=ax);
fig.show()
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 
                                           'Mlle', 'Lady', 'Mme', 
                                           'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 
                                           'Capt', 'Sir', 'Don', 'Rev'], 'Other')
sns.countplot(x='Title', hue='Survived', data=df_all);
df_all.drop('Name', axis=1, inplace=True)
df_all.drop('Ticket', axis=1, inplace=True)
df_all.head()
sex_dict = {'male':1, 'female':0}
df_all['Sex'] = df_all['Sex'].map(sex_dict)
embarked_dict = {'S':0, 'Q':1, 'C':2}
df_all['Embarked'] = df_all['Embarked'].map(embarked_dict)
title_dict = {'Mr':0, 'Miss/Mrs/Ms':1, 'Master':2, 'Other':3}
df_all['Title'] = df_all['Title'].map(title_dict)
df_all.head()
df_train, df_test = divide_df(df_all)
df_train = df_train.drop('PassengerId', axis=1)
df_train.head()
X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_STATE)

categorical_features = ['Embarked', 'Pclass', 'Title']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
logit = LogisticRegression(solver='lbfgs', max_iter=500,
                           random_state=RANDOM_STATE, n_jobs=-1)

parameters = {'C':[0.005, 0.01, 0.05, 0.1, 0.5, 1]}
logit_grid = GridSearchCV(logit, parameters)

logit_pipe = Pipeline([('preprocessor', preprocessor),
                       ('scaler', StandardScaler()), 
                       ('logit', logit_grid)])
logit_pipe.fit(X_train, y_train);
print('Best C: ', logit_grid.best_params_['C'])
print('Train accuracy:', accuracy_score(y_train, logit_pipe.predict(X_train)))
print('CV accuracy:', logit_grid.best_score_)
print('Test accuracy:', accuracy_score(y_valid, logit_pipe.predict(X_valid)))
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

parameters = {'n_estimators':[3, 5, 10, 30],
              'max_depth': range(1, 10)}

rf_grid = GridSearchCV(rf, parameters)

rf_pipe = Pipeline([('preprocessor', preprocessor), 
                    ('forest', rf_grid)])
rf_pipe.fit(X_train, y_train);
print('Best n_estimators: ', rf_grid.best_params_['n_estimators'])
print('Best max_depth: ', rf_grid.best_params_['max_depth'])
print('Train accuracy:', accuracy_score(y_train, rf_pipe.predict(X_train)))
print('CV accuracy:', rf_grid.best_score_)
print('Test accuracy:', accuracy_score(y_valid, rf_pipe.predict(X_valid)))
lgb_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE)
#lgb_clf.fit(X_train, y_train)
#accuracy_score(y_valid, lgb_clf.predict(X_valid))
param_grid = {'num_leaves': [7, 15, 31, 63], 
              'max_depth': [1, 2, 3, 4, 5, 6, -1]}

lgb_grid = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, 
                             cv=5, verbose=1, n_jobs=-1)

lgb_grid.fit(X_train, y_train, categorical_feature=categorical_features);
print('Best params: ', lgb_grid.best_params_)
print('Train accuracy:', accuracy_score(y_train, lgb_grid.predict(X_train)))
print('CV accuracy:', lgb_grid.best_score_)
print('Test accuracy:', accuracy_score(y_valid, lgb_grid.predict(X_valid)))
num_iterations = 2000
lgb_clf2 = lgb.LGBMClassifier(random_state=RANDOM_STATE, 
                              max_depth=lgb_grid.best_params_['max_depth'], 
                              num_leaves=lgb_grid.best_params_['num_leaves'], 
                              n_estimators=num_iterations,
                              n_jobs=-1)

param_grid2 = {'learning_rate': np.logspace(-4, 0, 10)}
lgb_grid2 = GridSearchCV(estimator=lgb_clf2, param_grid=param_grid2,
                               cv=5, verbose=1, n_jobs=4)

lgb_grid2.fit(X_train, y_train, categorical_feature=categorical_features)
print('Best params: ', lgb_grid2.best_params_)
print('Train accuracy:', accuracy_score(y_train, lgb_grid2.predict(X_train)))
print('CV accuracy:', lgb_grid2.best_score_)
print('Test accuracy:', accuracy_score(y_valid, lgb_grid2.predict(X_valid)))
final_lgb = lgb.LGBMClassifier(n_estimators=num_iterations,
                               max_depth=lgb_grid.best_params_['max_depth'], 
                               num_leaves=lgb_grid.best_params_['num_leaves'],
                               learning_rate=lgb_grid2.best_params_['learning_rate'],
                               n_jobs=-1, random_state=RANDOM_STATE)
final_lgb.fit(X, y, categorical_feature=categorical_features)
pd.DataFrame(final_lgb.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]
ids = df_test['PassengerId'].values
test_inputs = df_test.drop('PassengerId', axis=1)
predsTest = final_lgb.predict(test_inputs)
y = np.int32(predsTest > 0.5)
y = y.astype(int)

output = pd.DataFrame({'PassengerId': ids, 'Survived': y})
output.to_csv("submission.csv", index=False)