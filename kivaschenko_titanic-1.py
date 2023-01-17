# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
df = pd.concat([pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')])

print(df.info())
print(df.head())
print(df.head())
greeting = [x.split(',')[1].split('.')[0].strip() for x in df.Name.values]

df["greeting"] = greeting

print(df.head())
df['greeting'].value_counts()
sns.set_style("whitegrid")

sns.boxplot(x=df.greeting, y=df.Age)
df[df['greeting'] == 'Jonkheer']
df['greeting'] = df['greeting'].replace('Jonkheer', 'Mr')
df['greeting'] = df['greeting'].replace('Capt', 'Mr')

df['greeting'] = df['greeting'].replace('Dona', 'Mrs')

df['greeting'] = df['greeting'].replace('Don', 'Mr')

df['greeting'] = df['greeting'].replace('Sir', 'Mr')

df['greeting'] = df['greeting'].replace('Lady', 'Mrs')
df[df['greeting'] == 'the Countess']
df['greeting'] = df['greeting'].replace('the Countess', 'Miss')
df[df['greeting'] == 'Mme']
df['greeting'] = df['greeting'].replace('Mme', 'Miss')
df['greeting'].value_counts()
df[df['greeting'] == 'Mlle']
df['greeting'] = df['greeting'].replace('Mlle', 'Miss')
df[df['greeting'] == 'Ms']
df['greeting'] = df['greeting'].replace('Ms', 'Miss')
df[df['greeting'] == 'Major']
df[df['greeting'] == 'Major']
df['greeting'] = df['greeting'].replace('Major', 'Mr')
table = df.pivot_table(values='Age', \

                       index=['greeting'], \

                       columns=['Pclass', 'Sex'], \

                       aggfunc=np.median)

print(table)
df[df.Fare.isnull()]
df[df.Embarked.isnull()]
df.sort_values(by='Ticket')[50:70]
df['Embarked'].fillna('S', inplace=True)
df.info()
df[df.Fare.isnull()]
df.sort_values(by='Ticket', ascending=False)[440:460]
fare_mean = df[(df.Age>55)&(df.Pclass==3)&(df.Sex=='male')]['Fare'].mean()

print(fare_mean)
df["Fare"].fillna(fare_mean, inplace=True)
df.info()
df["Sex"].value_counts()
sex_dummies = pd.get_dummies(df["Sex"])

print(sex_dummies.info())

print(sex_dummies[:5])
df['female'] = sex_dummies['female']

df.drop('Sex', axis=1, inplace=True)
df.tail()
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
df.tail()
df.Embarked.value_counts()
df.greeting.value_counts()
def get_dummie(data, column):

    """Convert to binar number of categories"""

    

    df_dummie = pd.get_dummies(df[column][:], prefix=column)

    df_dummie = pd.concat([df[:],df_dummie[:]], axis=1)

    

    return (df_dummie)
x = df.loc[:,:]

df = get_dummie(x, "Embarked")
df.info()
x = df.loc[:,:]

df = get_dummie(x, 'greeting')
df.info()
df.drop(['Embarked', 'greeting'], axis=1, inplace=True)
df.info()
train_age_df = df[df.Age.notnull()]

print(train_age_df.info())
test_age_df = df[df.Age.isnull()]

print(test_age_df.shape)
train_age_df.Age = (train_age_df.loc[:,'Age']+1).apply(np.log)
y = train_age_df.Age.values

print("Shape of target_age is:{}".format(y.shape))

X = train_age_df.drop(['Age', 'PassengerId', 'Survived'], axis=1)

print("Shape of train_age is:{}".format(X.shape))

test_age = test_age_df.drop(['Age', 'PassengerId', 'Survived'], axis=1)

print("Shape of test_age is:{}".format(test_age.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_pred_scaled = scaler.transform(test_age)
print("Shape X_scaled: {0}. Shape y: {1}. Shape X_pred_scaled : {2}"\

      .format(X_scaled.shape, y.shape, X_pred_scaled.shape))

print("X_scaled[:10]: {0}.\n\n\n y[:10]:{1}.\n\n\n X_pred_scaled[:10]:{2}" \

    .format(X_scaled[:10], y[:10], X_pred_scaled[:10]))
param_grid = {'n_estimators': [n for n in range(10, 110, 10)],

              'max_depth'   : [d for d in range(2,7)],

              'max_features': [f for f in range(2, 6)]}

print("Parameter grid:\n{}".format(param_grid))
grid_search = GridSearchCV(RandomForestRegressor(random_state=42, 

                                                n_jobs=-1), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=21)
grid_search.fit(X_train, y_train)

print("Test set score: {:.5f}".format(grid_search.score(X_test, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))

print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))

print("Best estimator:\n{}".format(grid_search.best_estimator_))
regressor = grid_search.best_estimator_
regressor.fit(X_train, y_train)

print("Accuracy on training set: {:.5f}".format(regressor.score(X_train, y_train)))

print("Accuracy on test set: {:.5f}".format(regressor.score(X_test, y_test)))
kfold = KFold(n_splits=5)

scores = cross_val_score(regressor, X, y, cv=kfold)

print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.5f}".format(scores.mean()))
predicted_age = regressor.predict(X_pred_scaled)

print(predicted_age[:20])
predicted_age = np.exp(predicted_age[:]) - 1

predicted_age = np.round(predicted_age, 2)

print(predicted_age[:20])
predicted_age.shape
df[df.Age.isnull()].values.shape
test_age_df["Age"] = predicted_age

test_age_df.info()
test_age_df.info()
train_age_df.info()
df = pd.concat([train_age_df, test_age_df])

df.info()
train = df[(df.Survived == 1) | (df.Survived == 0)]

train.info()
survived = train['Survived'][:].values

print(survived.shape)

print(survived[:10])
train.drop(['PassengerId', 'Survived'], axis=1, inplace=True)

print(train.info())
test = df[(df.Survived != 1) & (df.Survived != 0)]

test.info()
PassId = test.PassengerId[:]

print(type(PassId))

print(PassId[:5])
test.drop(['PassengerId', 'Survived'], axis=1, inplace=True)

print(test.info())
X = train.values

y = survived

X_pred = test.values
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_pred_scaled = scaler.transform(X_pred)



print("Shape X_scaled: {0}. Shape y: {1}. Shape X_pred_scaled : {2}"\

      .format(X_scaled.shape, y.shape, X_pred_scaled.shape))

print("X_scaled[:10]: {0}.\n\n\n y[:10]:{1}.\n\n\n X_pred_scaled[:10]:{2}" \

    .format(X_scaled[:10], y[:10], X_pred_scaled[:10]))
param_grid = {'n_estimators': [n for n in range(10, 110, 10)],

              'max_depth'   : [d for d in range(2,5)],

              'max_features': [f for f in range(2, 9)]}

print("Parameter grid:\n{}".format(param_grid))
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, 

                                                n_jobs=-1), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=21)

grid_search.fit(X_train, y_train)

print("Test set score: {:.5f}".format(grid_search.score(X_test, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))

print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))

print("Best estimator:\n{}".format(grid_search.best_estimator_))
forest = grid_search.best_estimator_



forest.fit(X_train, y_train)

print("Accuracy on training set: {:.5f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.5f}".format(forest.score(X_test, y_test)))
kfold = KFold(n_splits=5)

scores = cross_val_score(forest, X, y, cv=kfold)

print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.5f}".format(scores.mean()))
submission = pd.read_csv('../input/genderclassmodel.csv')

submission.iloc[:, 1] = forest.predict(X_pred_scaled)

submission.to_csv('random_forest_clf_titanic_subm.csv', index=False)