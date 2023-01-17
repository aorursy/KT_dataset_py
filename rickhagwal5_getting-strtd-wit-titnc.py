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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data
women = train_data.loc[train_data.Sex=='female']["Survived"]

women

rate_women = sum(women)/len(women)

rate_women
men = train_data.loc[train_data.Sex=='male']["Survived"]

men

rate_men=sum(men)/len(men)

rate_men
train_data.dtypes
test_data.dtypes
import pandas as pd

from sklearn.model_selection import train_test_split



# X_full- train_data, X_test_full- test_Data

# Read the data



# Remove rows with missing target, separate target from predictors

# train_data.dropna(axis=0, subset=['Survived'], inplace=True)

# y = train_data.Survived

# train_data.drop(['Survived'], axis=1, inplace=True)



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



# To keep things simple, we'll use only numerical predictors

# X = train_data.select_dtypes(exclude=['object'])

# X_test = test_data.select_dtypes(exclude=['object'])



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_test
X.dtypes
X_test.dtypes
# Shape of training data (num_rows, num_columns)

print(X_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Shape of training data (num_rows, num_columns)

print(X_test.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_test.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

    model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

    #model = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=5)

    #model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    preds_test = model.predict(X_test)

    #var = int(preds_test)

    print(np.around(preds_test).astype(int))

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': np.around(preds_test).astype(int)})

    output.to_csv('submission_t.csv', index=False)

    return mean_absolute_error(y_valid, preds)
test_data
print(score_dataset(X_train, X_valid, y_train, y_valid))
out = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': np.around(preds_test).astype(int), 'Sex': test_data.Sex})

out
women_test = out.loc[out.Sex=='female']["Survived"]

women_test

rate_women_test = sum(women_test)/len(women_test)

rate_women_test
men_test = out.loc[out.Sex=='male']["Survived"]

men_test

rate_men_test = sum(men_test)/len(men_test)

rate_men_test