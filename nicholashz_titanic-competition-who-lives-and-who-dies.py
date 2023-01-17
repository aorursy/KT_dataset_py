# Packages for the data
import pandas as pd
import numpy as np

# Package for visualization
import seaborn as sns

# Packages for machine learning classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
# Read the data into dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Put all data together so we can wrangle all the data at the same time
all_data = pd.concat(objs=[train_df, test_df], axis=0, sort=True).reset_index(drop=True)

all_data.head()
train_df.drop(labels='Ticket', axis='columns', inplace=True)
test_df.drop(labels='Ticket', axis='columns', inplace=True)
# Find out what columns have null values
all_data.isnull().sum()
for df in [train_df, test_df]:
    # Fill NaN values for Embarked and Fare
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

train_df.isnull().sum()
for df in [train_df, test_df]:
    df.loc[df['Cabin'].isnull(), 'HasCabin'] = 0
    df.loc[df['Cabin'].notnull(), 'HasCabin'] = 1
    
    df.drop(labels='Cabin', axis='columns', inplace=True)

pd.crosstab(train_df['Sex'], train_df['HasCabin'])
sns.catplot(x='Sex', y='Survived', kind='bar', data=train_df)
sns.catplot(x='Embarked', y='Survived', kind='bar', data=train_df)
train_dummies = pd.get_dummies(data=train_df[['Sex', 'Embarked']])
train_df = pd.concat([train_df, train_dummies], axis=1)

test_dummies = pd.get_dummies(data=test_df[['Sex', 'Embarked']])
test_df = pd.concat([test_df, test_dummies], axis=1)

for df in [train_df, test_df]:
    df.drop(labels=['Sex', 'Embarked'], axis='columns', inplace=True)

train_df.head()
for df in [train_df, test_df]:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Pclass'], train_df['Title'])
for df in [train_df, test_df]:
    # Combine odd or repeated titles
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', \
                                       'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona'], 'Other')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Drop the names
    df.drop(labels='Name', axis='columns', inplace=True)

sns.catplot(x='Title', y='Survived', kind='bar', data=train_df)
train_dummies = pd.get_dummies(data=train_df['Title'], prefix='Title')
train_df = pd.concat([train_df, train_dummies], axis=1)

test_dummies = pd.get_dummies(data=test_df['Title'], prefix='Title')
test_df = pd.concat([test_df, test_dummies], axis=1)

for df in [train_df, test_df]:
    df.drop(labels='Title', axis='columns', inplace=True)

train_df.head()
for df in [train_df, test_df]:
    # Replace SibSp and Parch with a single FamilySize column
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.drop(labels=['SibSp', 'Parch'], axis='columns', inplace=True)

train_df[['FamilySize', 'Survived']].groupby('FamilySize').mean()
for df in [train_df, test_df]:
    df.loc[df['FamilySize'] == 1, 'FamilySize'] = 0
    df.loc[(df['FamilySize'] == 2) | (df['FamilySize'] == 3), 'FamilySize'] = 1
    df.loc[df['FamilySize'] == 4, 'FamilySize'] = 2
    df.loc[df['FamilySize'] > 4, 'FamilySize'] = 3

sns.catplot(x='FamilySize', y='Survived', kind='bar', data=train_df)
corr = train_df[['Age', 'Pclass', 'HasCabin', 'FamilySize']].corr()
sns.heatmap(corr, cmap=sns.color_palette('coolwarm', 7), center=0)
corr
for df in [train_df, test_df]:
    age_medians = np.zeros(shape=(3, 4))
    for pclass in range(age_medians.shape[0]):
        for familysize in range(age_medians.shape[1]):
            age_medians[pclass][familysize] = df.loc[(df['Pclass'] == pclass+1) & \
                                                     (df['FamilySize'] == familysize),
                                                     'Age'].median()
            df.loc[(df['Age'].isnull()) & \
                   (df['Pclass'] == pclass+1) & \
                   (df['FamilySize'] == familysize),
                    'Age'] = age_medians[pclass][familysize]

# Double check that we have no more null values
train_df.isnull().sum()
train_df['AgeBand'] = pd.cut(train_df['Age'], 10)
train_df[['AgeBand', 'Survived']].groupby('AgeBand').mean()
train_df.drop(labels='AgeBand', axis='columns', inplace=True)

for df in [train_df, test_df]:
    df.loc[(df['Age'] > 0) & (df['Age'] <= 8), 'AgeGroup'] = 0
    df.loc[(df['Age'] > 8) & (df['Age'] <= 32), 'AgeGroup'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 56), 'AgeGroup'] = 2
    df.loc[df['Age'] > 56, 'AgeGroup'] = 3

    df.drop(labels='Age', axis='columns', inplace=True)
    
sns.catplot(x='AgeGroup', y='Survived', kind='bar', data=train_df)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby('FareBand').mean()
train_df.drop(labels='FareBand', axis='columns', inplace=True)

for df in [train_df, test_df]:
    df.loc[df['Fare'] <= 7.91, 'FareGroup'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.45), 'FareGroup'] = 1
    df.loc[(df['Fare'] > 14.45) & (df['Fare'] <= 31), 'FareGroup'] = 2
    df.loc[df['Fare'] > 31, 'FareGroup'] = 3
    
    df.drop(labels='Fare', axis='columns', inplace=True)
    
sns.catplot(x='FareGroup', y='Survived', kind='bar', data=train_df)
train_df.head()
Ytrain = train_df['Survived']
Xtrain = train_df.drop(columns=['PassengerId', 'Survived'])

RFC = RandomForestClassifier()
Ada = AdaBoostClassifier()
KNN = KNeighborsClassifier()
classifiers = [RFC, Ada, KNN]
clf_names = ['Random Forest', 'AdaBoost', 'K Nearest Neighbors']
# Use kfold as our cross validation
kfold = StratifiedKFold(n_splits=10)

# Set grid search parameter settings
rfc_param_grid = {'max_depth': [None],
                 'max_features': [1, 4, 8],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 5, 10],
                 'bootstrap': [False],
                 'n_estimators': [100, 300, 500],
                 'criterion': ['gini']}
ada_param_grid = {'n_estimators': [25, 50, 100, 200],
                 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
knn_param_grid = {'n_neighbors': [5, 10, 20, 30, 50, 100],
                  'weights': ['uniform', 'distance'],
                 'leaf_size': [5, 10, 20, 30, 50, 100]}
param_grids = [rfc_param_grid, ada_param_grid, knn_param_grid]

# Perform grid searches to get estimators with the optimal settings
grid_searches = []
for i in range(len(classifiers)):
    grid_searches.append(GridSearchCV(estimator=classifiers[i], param_grid=param_grids[i], 
                                      n_jobs=4, cv=kfold, verbose=1))
# Train the models
best_scores = []
for i in range(len(grid_searches)):
    grid_searches[i].fit(Xtrain, Ytrain)
    best_scores.append(grid_searches[i].best_score_)
# Best scores
for i in range(len(best_scores)):
    print(clf_names[i] + ": " + str(best_scores[i]))
# Make predictions
Xtest = test_df.drop(columns='PassengerId', axis='columns')
predictions = grid_searches[0].predict(Xtest)

# Write predictions to output csv
pred_df = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                        'Survived': predictions})
pred_df.to_csv('predictions.csv', index=False)

print("Done writing to csv")