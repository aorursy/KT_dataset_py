# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import test/train data from the given files.

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
#What does the test data look like?

train_data.head()
# Create X_train data

full_train = train_data.drop(['Name', 'Parch', 'Ticket'], axis = 1)

full_train.head()
# Explore the training data - what does it look like?

full_train.info()
# Fill in NaN values. Embarked with most frequent

full_train = full_train[pd.notnull(full_train['Embarked'])]
# Deal with missing age values

full_train = full_train[pd.notnull(full_train['Age'])]



full_train.info()
# Convert Cabin entries to binary. 0 if Nan, 1 if present

full_train['Cabin'] = full_train['Cabin'].where(full_train['Cabin'].isna(), 1)

full_train['Cabin'] = full_train['Cabin'].fillna(0)

full_train['Cabin'].head()
# Use OneHotEncoder to encode the Sex and Emarked column

from sklearn.preprocessing import OneHotEncoder



s = (full_train.dtypes == 'object')

object_cols = list(s[s].index)



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols = pd.DataFrame(OH_encoder.fit_transform(full_train[object_cols]))



# One-hot encoding removed index; put it back

OH_cols.index = full_train.index



# Remove categorical columns (will replace with one-hot encoding)

num = full_train.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH = pd.concat([num, OH_cols], axis=1)



OH.columns
# Now it has been cleaned, split the training data provided into training and validation data to apply and test different methods

# Break off validation set from training data

from sklearn.model_selection import train_test_split

X = OH.drop('Survived', axis=1)

y = OH['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
# Create a RandomForest model and apply to the test data, then apply to validation data nd check MAE.

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



model = RandomForestRegressor(n_estimators=1500, random_state=0)

model.fit(X_train, y_train)

preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)

print (mae)
# Clean test data provided as with the training set

test_data = test_data.drop(['Name', 'Parch','Ticket'], axis = 1)

#test_data = test_data[pd.notnull(test_data['Embarked'])]

#test_data = test_data[pd.notnull(test_data['Age'])]



test_data['Cabin'] = test_data['Cabin'].where(test_data['Cabin'].isna(), 1)

test_data['Cabin'] = test_data['Cabin'].fillna(0)



OH_cols = pd.DataFrame(OH_encoder.fit_transform(test_data[object_cols]))



# One-hot encoding removed index; put it back

OH_cols.index = test_data.index



# Remove categorical columns (will replace with one-hot encoding)

num = test_data.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_test_data = pd.concat([num, OH_cols], axis=1)



# Replace missing values with mean

OH_test_data.fillna(OH_test_data.mean(), inplace = True)
# Apply model to the cleaned test data, to predict survival of Titanic passengers to submit to competition.

prediction = model.predict(OH_test_data)

test_data['Survived'] = prediction



submission = test_data[['PassengerId','Survived']]

submission['Survived'].values[submission['Survived'] >= 0.7] = 1

submission['Survived'].values[submission['Survived'] < 0.7] = 0



submission.Survived = submission.Survived.astype(int)



submission.to_csv('titanic_randomforest.csv', index=False)



submission.info()