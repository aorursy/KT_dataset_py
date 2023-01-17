# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
Submit=pd.read_csv("../input/sample_submission.csv")

from sklearn.ensemble import RandomForestClassifier
train_x=train.drop('label',axis=1)
train_y=train['label']

model_rf=RandomForestClassifier(random_state=100)
model_rf.fit(train_x,train_y)

test_pred=model_rf.predict(test)
df_rf=pd.DataFrame({'ImageId':list(range(1,len(test_pred)+1)),'Label':test_pred})
df_rf.to_csv('Assignment.csv',index=False)

# Any results you write to the current directory are saved as output.
