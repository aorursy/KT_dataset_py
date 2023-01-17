import numpy as np

import pandas as pd
train_data = pd.read_csv('../input/train.csv')
len(train_data)
import sklearn.linear_model

model = sklearn.linear_model.LogisticRegression()

X = train_data.drop('target', axis = 1).as_matrix()

Y = train_data.as_matrix(columns = ['target']).flatten()

model.fit(X, Y)
test_data = pd.read_csv('../input/test.csv')

predictions = pd.DataFrame(columns = ['id', 'target'])

predictions['id'] = test_data['id']

XP = test_data.as_matrix()

predictions['target'] = model.predict(XP)

predictions.to_csv('submission.csv', index = False)