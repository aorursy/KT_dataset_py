# Simple model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier #  Decision Trees classifier model

train = pd.read_csv('../input/dice-dataset/dice_train.csv')

# it is true/false column
# 0 - bad dice, 1 - good dice
y = train.isTruthful

# we get first 6 tries
X = train[['try0', 'try1', 'try2',  'try3', 'try4', 'try5']]

model = DecisionTreeClassifier()

model.fit(X, y)

# fake dice - it is not possible that all calls are 6 points
print(model.predict([[6,6,6,6,6,6]]))

# good dice each try is different
print(model.predict([[1,2,3,4,5,6]]))
