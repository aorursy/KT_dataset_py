# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_sample = pd.read_csv('../input/train_sample.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

np.random.seed(0)
train_sample.head()
train_sample.info()
train_sample.describe(include= 'all')
test.head()
predictors = ['ip', 'app', 'device', 'os', 'channel' ]
y = train_sample['is_attributed']
x_train = train_sample[predictors]
x_test = test[predictors]
my_pipeline = make_pipeline(Imputer(), RandomForestClassifier())


scores = cross_val_score(my_pipeline, x_train, y, scoring='roc_auc', cv=5)
print(scores)
my_pipeline.fit(x_train, y)
prediction = my_pipeline.predict(x_test)
print(prediction)
my_submission = pd.DataFrame({'click_id': test.click_id, 'is_attributed': prediction})
my_submission.to_csv('submission.csv', index=False)
