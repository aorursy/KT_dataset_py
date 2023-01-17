import pandas as pd
import numpy as np

from sklearn import metrics, base
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold

test = pd.read_csv("../input/klps-creditscring-challenge-for-students/test.csv")
train = pd.read_csv("../input/klps-creditscring-challenge-for-students/test.csv")
submission = test[['id']]
lgb = pd.read_csv('../input/lgb-kalapa/submission.csv')
cat = pd.read_csv('../input/cat-kalapa/submission.csv')
submission['label'] = lgb['label']*0.5 + cat['label']*0.5
submission.to_csv('submission.csv',index=False)
submission['label'].hist()