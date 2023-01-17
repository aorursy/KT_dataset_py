import numpy as np
import pandas as pd
import os
import xgboost as xgb
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
y = train.AdoptionSpeed.values
train = train.drop(['Name', 'RescuerID', 'Description', 'PetID', 'AdoptionSpeed'], axis=1).values
test = test.drop(['Name', 'RescuerID', 'Description', 'PetID'], axis=1).values
clf = xgb.XGBClassifier(n_estimators=500, nthread=-1, max_depth=8, learning_rate=0.015)
clf.fit(train, y)
preds = clf.predict(test)
sample = pd.read_csv('../input/test/sample_submission.csv')
sample.AdoptionSpeed = preds
sample.to_csv('submission.csv', index=False)
