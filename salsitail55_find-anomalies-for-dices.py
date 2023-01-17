# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Read the data
train = pd.read_csv('../input/dice_test(1).csv')

# pull data into target (y) and predictors (X)
y = train.isTruthful
predictor_cols = ['try0', 'try1', 'try2', 'try3', 'try4', 'try5', 'try6', 'try7', 'try8', 'try9', 'try10', 'try11', 'try12', 'try13']

# Create training predictors data
X = train.drop(columns=['isTruthful']) #train[predictor_cols]

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
dice_model = DecisionTreeRegressor()
# Fit model
dice_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = dice_model.predict(val_X)

def roundValue(x): 
    if (x < 0.75):
        return 0
    else:
        return 1

print(list(map(roundValue, val_predictions)))
#print (val_y)
print(mean_absolute_error(val_y, val_predictions))

#print(dice_model.predict([[1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]]))



#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
#my_submission.to_csv('submission.csv', index=False)


# Any results you write to the current directory are saved as output.
