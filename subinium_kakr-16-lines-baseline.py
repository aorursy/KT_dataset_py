import numpy as np 

import pandas as pd

from category_encoders.ordinal import OrdinalEncoder

from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('/kaggle/input/kakr-4th-competition/train.csv')

test = pd.read_csv('/kaggle/input/kakr-4th-competition/test.csv')

sample_submission = pd.read_csv('/kaggle/input/kakr-4th-competition/sample_submission.csv')

target = train['income'] != '<=50K'

train.drop(['income'], axis=1, inplace=True)

LE_encoder = OrdinalEncoder(list(train.columns))

train_le = LE_encoder.fit_transform(train, target)

test_le = LE_encoder.transform(test)

lr_clf = DecisionTreeClassifier()

lr_clf.fit(train_le, target)

sample_submission['prediction'] = lr_clf.predict(test_le).astype(int)

sample_submission.to_csv('submission.csv', index=False)