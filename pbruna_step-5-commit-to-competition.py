# How to commit results to competition

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('../input/dice-dataset/dice_train.csv')
y = train.isTruthful
X = train.drop(columns=['Id', 'isTruthful'])

# train model on train dataset
model = GradientBoostingClassifier()
model.fit(X, y)

# load test dataset
test = pd.read_csv('../input/dice-dataset/dice_test.csv')

# competition test dataset
X_test = test.drop(columns=['Id', 'isTruthful'])

# computate result with help of trained model
predicted_isTruthful = model.predict(X_test)

# create submission as frame with two columns
my_submission = pd.DataFrame({'Id': test.Id, 'isTruthful': predicted_isTruthful})

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

