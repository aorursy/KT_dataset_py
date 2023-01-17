# accuracy score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/dice-dataset/dice_train.csv')
y = train.isTruthful
X = train[['try0', 'try1', 'try2', 'try3', 'try4', 'try5']]

model = DecisionTreeClassifier()
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# train model on training subset
model.fit(train_X, train_y)

# get predicted result for validation subset
predicted_isTruthful = model.predict(val_X)

# show mean error for accuracy
print(accuracy_score(val_y, predicted_isTruthful))



