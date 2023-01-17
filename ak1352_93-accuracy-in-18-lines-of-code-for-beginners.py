import numpy as np

import pandas as pd

from xgboost import XGBClassifier
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submit_sample = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

print('Shape of Training Data: ', train.shape)

print('Shape of Testing Data: ', test.shape)
X = train.drop(columns=['label'])

Y = train['label']
from xgboost import XGBClassifier
xgb = XGBClassifier(n_jobs=5).fit(X,Y)
y_pred = xgb.predict(test)
result = pd.DataFrame()

result['Label'] = y_pred

result['ImageId'] = submit_sample.ImageId

result.set_index('ImageId')
result.to_csv('digit_recognizer_result.csv')