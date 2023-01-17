# import libraries

import numpy as np 

import pandas as pd 

import re



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
# Load in the train and test datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access

PassengerId = test['PassengerId']



train.head(3)
full_data = [train, test]



# Some features of my own that I have added in

# Gives the length of the name

train['Name_length'] = train['Name'].apply(len)

test['Name_length'] = test['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
print(train.info())

print(train.head())
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)
train.head(3)
# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models

y = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

X = train.values # Creates an array of the train data

X_pred = test.values # Creats an array of the test data
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_pred_scaled = scaler.transform(X_pred)
print("Shape X_scaled: {0}. Shape y: {1}. Shape X_pred_scaled : {2}"\

      .format(X_scaled.shape, y.shape, X_pred_scaled.shape))

print("X_scaled[:10]: {0}.\n\n\n y[:10]:{1}.\n\n\n X_pred_scaled[:10]:{2}" \

    .format(X_scaled[:10], y[:10], X_pred_scaled[:10]))
param_grid = {'n_estimators': [n for n in range(20, 200, 20)],

              'max_depth'   : [d for d in range(3,7)],

              'max_features': [f for f in range(2, 7)]}

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



# forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#             max_depth=7, max_features=4, max_leaf_nodes=None,

#             min_impurity_decrease=0.0, min_impurity_split=None,

#             min_samples_leaf=1, min_samples_split=2,

#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,

#             oob_score=False, random_state=42, verbose=0, warm_start=False)

forest.fit(X_train, y_train)

print("Accuracy on training set: {:.5f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.5f}".format(forest.score(X_test, y_test)))
# To check the scores by cross-validation class

kfold = KFold(n_splits=5)

scores = cross_val_score(forest, X, y, cv=kfold)

print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.5f}".format(scores.mean()))
# To write the results in file

submission = pd.read_csv('../input/genderclassmodel.csv')

submission.iloc[:, 1] = forest.predict(X_pred_scaled)

submission.to_csv('random_forest_clf_titanic_subm.csv', index=False)