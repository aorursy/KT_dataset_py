# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from sklearn import linear_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import genfromtxt, savetxt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('../input/train.csv')
train_set.head()
train_set_X = train_set.iloc[:, 1:]
train_set_Y = train_set.iloc[:, 0]

test_set_X = pd.read_csv('../input/test.csv')
model = linear_model.SGDClassifier()
model.fit(train_set_X, train_set_Y)
print(os.listdir("../input"))
train_set.shape
test_set_X.shape
prediction = model.predict(test_set_X)
prediction
sm = pd.DataFrame({'ImageId': range(1, len(prediction) + 1), 'Label': prediction})
sm.to_csv('my_submission.csv', index=False)

