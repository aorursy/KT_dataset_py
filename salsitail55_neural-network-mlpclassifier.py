# accuracy score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('../input/dice-dataset/dice_train.csv')
y = train.isTruthful
columns=['try0', 'try1', 'try2', 'try3', 'try4', 'try5', 'try6', 'try7', 'try8', 'try9', 'try10', 'try11']

X = train[columns]
#X = train.drop(columns=['Id', 'isTruthful'])

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

model2 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(100, 2), random_state=1)

model2.fit(train_X, train_y)

predicted_isTruthful2 = model2.predict(val_X)

# show mean error for accuracy
print(accuracy_score(val_y, predicted_isTruthful2))


