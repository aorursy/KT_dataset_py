# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv('../input/sample_submission.csv')
train_x = train.drop('label', axis=1)
train_y = train['label']
model_rf = RandomForestClassifier(random_state=100, n_estimators=800)

model_rf.fit(train_x, train_y)
pred = model_rf.predict(test)
df_pred_rf = pd.DataFrame({'ImageId': list(range(1, len(pred)+1)), 'Label': pred})
df_pred_rf.to_csv('submission.csv', index = False)
# print(df_pred_rf)
