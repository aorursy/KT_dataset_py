# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
#Copy the data and fill missing values in age with the average age

train_set = train_data.copy()
train_set['Age'] = train_set['Age'].fillna(value = train_set['Age'].mean(axis=0))

test_set = test_data.copy()
test_set['Age'] = test_set['Age'].fillna(value = test_set['Age'].mean(axis=0))
#Fill Fare Missing Values with Average Fare
train_set['Fare'] = train_set['Fare'].fillna(train_set['Fare'].mean())
test_set['Fare'] = test_set['Fare'].fillna(test_set['Fare'].mean())
#Split the Cabin into Cabin Category and Cabin Number
train_set['Cabin Cat'] = train_set.Cabin.str[0]
train_set['Cabin No'] = train_set.Cabin.str[1:]
train_set = train_set.drop(['Cabin'], axis=1)
train_set[['Cabin Cat', 'Cabin No']] = train_set[['Cabin Cat', 'Cabin No']].fillna('O')

test_set['Cabin Cat'] = test_set.Cabin.str[0]
test_set['Cabin No'] = test_set.Cabin.str[1:]
test_set = test_set.drop(['Cabin'], axis=1)
test_set[['Cabin Cat', 'Cabin No']] = test_set[['Cabin Cat', 'Cabin No']].fillna('O')
#Drop the features which will not be used for the model
train_set = train_set.drop(['Cabin No'], axis=1)
train_set = train_set.drop(['Name'], axis=1)
train_set = train_set.drop(['Ticket'], axis=1)

test_set = test_set.drop(['Cabin No'], axis=1)
test_set = test_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Ticket'], axis=1)
train_set.head()
test_set.head()
#Get the y target for the training data
y_train = np.array(train_set.Survived)
y_train.shape
#Get the taining features
X_features = train_set.drop(['Survived'], axis=1)
X_features.head()
#Assign the test features
test_features = test_set
test_features.head()
#Fill missing values in the features
X_features['Embarked'] = X_features['Embarked'].fillna(method='ffill')
test_features['Embarked'] = test_features['Embarked'].fillna(method='ffill')
#Label Encode the 'Embarked', 'Sex' and 'Cabin Cat' Variables
X_features['Embarked'] = LabelEncoder().fit_transform(X_features['Embarked'])
test_features['Embarked'] = LabelEncoder().fit_transform(test_features['Embarked'])

X_features['Cabin Cat'] = LabelEncoder().fit_transform(X_features['Cabin Cat'])
test_features['Cabin Cat'] = LabelEncoder().fit_transform(test_features['Cabin Cat'])

X_features['Sex'] = LabelEncoder().fit_transform(X_features['Sex'])
test_features['Sex'] = LabelEncoder().fit_transform(test_features['Sex'])
X_features.head()
test_features.head()
#Convert the features to 'float'
X_features = X_features.astype(float)
test_features = test_features.astype(float)
X_features.dtypes
test_features.dtypes
#Scale the Features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_features)
X_test = scaler.fit_transform(test_features)
X_train.shape
X_test.shape
#Tune Hyperparameters 

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
model = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 100, cv = 3, 
                               verbose=2, random_state=42, 
                               n_jobs = -1)# Fit the random search model
#Fit the Model
model.fit(X_train, y_train)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
output.head()
