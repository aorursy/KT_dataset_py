# accuracy score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('../input/dice-dataset/dice_train.csv')
y = train.isTruthful
X = train[['try0', 'try1', 'try2', 'try3', 'try4', 'try5']]

model = DecisionTreeClassifier()
model.fit(X, y)

# get predicted result
predicted_isTruthful = model.predict(X)

# show mean error
print(accuracy_score(y, predicted_isTruthful))

