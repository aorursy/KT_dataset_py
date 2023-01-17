# How to commit results to competition

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeRegressor #  Decision Trees regression model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/dice-dataset/dice_train.csv')
y = train.isTruthful
columns=['try0', 'try1', 'try2', 'try3', 'try4', 'try5']
X = train[columns]

model = DecisionTreeRegressor()
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# train model on training subset
model.fit(X, y)

def roundValue(x): 
    if (x < 0.5):
        return 0
    else:
        return 1

# load test dataset
test = pd.read_csv('../input/dice-dataset/dice_test.csv')
X_test = test[columns]
predicted_isTruthful = model.predict(X_test)
predicted_isTruthful = list(map(roundValue, predicted_isTruthful))

my_submission = pd.DataFrame({'Id': test.Id, 'isTruthful': predicted_isTruthful})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

