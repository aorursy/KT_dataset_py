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
from sklearn.model_selection import train_test_split



# Read the data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



#separting the target from predictors and remocing rows with no predictor from the train data

train_data.dropna(axis=0, subset=['Survived'], inplace=True)

y = train_data.Survived            

train_data.drop(['Survived'], axis=1, inplace=True)

#figuring out missing values here

missing_val_count_by_column = (train_data.isnull().sum())

#print(missing_val_count_by_column[missing_val_count_by_column > 0])

train_data.Cabin.value_counts()



#Cabin doesn't seem to be helpful, and most of its values are missing. Maybe a source of data leakage. I should have thought of that earlier.

reduced_train_data = train_data.drop('Cabin', axis=1)

reduced_test_data = test_data.drop('Cabin', axis=1)



#Just curious about the ages in an attempt to figure out what strategy to use for the imputer

reduced_train_data.Age.value_counts()



#renaming to X and such because I should have done this earlier

X_full=reduced_train_data

X_test=reduced_test_data



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,random_state=0)

#Getting a handle on the categorical data

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Select categorical columns with relatively low cardinality, though this includes only Sex and Embarked

categorical_cols = [cname for cname in X_train.columns if

                    X_train[cname].nunique() < 10 and 

                    X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if 

                X_train[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train_reduced = X_train[my_cols].copy()

X_valid_reduced = X_valid[my_cols].copy()

X_test_reduced = X_test[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestClassifier(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train_reduced, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid_reduced)

preds
#Now that the random forest model has been set up above, let's see how bad things are

print('MAE:', mean_absolute_error(y_valid, preds))
#Still working with a random forest model, and doing some cross validation

from sklearn.model_selection import cross_val_score

def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    

    preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

    

    # Define model

    model = RandomForestClassifier(n_estimators, random_state=0)



    # Bundle preprocessing and modeling code in a pipeline

    clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])





    

    scores = -1 * cross_val_score(clf, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')

    

    return scores.mean()
#Noting that cross-validation requires me to un-split the data, we fix the full training data in teh same way that we fixed

#the original validation 

# Select categorical columns with relatively low cardinality, though this includes only Sex and Embarked

categorical_cols = [cname for cname in train_data.columns if

                    train_data[cname].nunique() < 10 and 

                    train_data[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in train_data.columns if 

                train_data[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X = train_data[my_cols].copy()

X_test_red = X_test[my_cols].copy()
y.dtype
# An attempt to optimize the estimators parameter

estimators=[50, 100, 150, 200, 250, 300, 350, 400]



results = { i : get_score(i) for i in estimators} 

import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
#May be worth checking a bit more



estimators=[150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]



results = { i : get_score(i) for i in estimators} 
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
model = RandomForestClassifier(n_estimators=300, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X, y)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid_reduced)

print('MAE:', mean_absolute_error(y_valid, preds))
preds_test = clf.predict(X_test_red)
def prob_to_int(prob):

    if prob <.5:

        return 0

    else:

        return 1



predictions=[prob_to_int(n) for n in preds_test]

print(preds_test)

print(predictions)
# Save test predictions to file

output = pd.DataFrame({'PassengerId': X_test_red.PassengerId,

                       'Survived': preds_test})

output.to_csv('submission.csv', index=False)