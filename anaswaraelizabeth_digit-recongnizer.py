# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.ensemble import RandomForestClassifier

train1 = pd.read_csv('../input/train.csv')
validate = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
train1_x=train1.drop('label',axis=1)
train1_y=train1['label']
model_rf = RandomForestClassifier(random_state=100, n_estimators=800)

model_rf.fit(train1_x, train1_y)
pred = model_rf.predict(validate)
df_predict_rf = pd.DataFrame({'ImageId': list(range(1, len(pred)+1)), 'Label': pred})
df_predict_rf.to_csv('submission.csv', index = False)
# print(df_pred_rf)
