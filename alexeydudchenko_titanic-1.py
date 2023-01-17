# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = train_df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_y = train.iloc[:,1]
train_x = train[["Pclass", "Sex", "Age", "SibSp"]]
test_x = test[["Pclass", "Sex", "Age", "SibSp"]]
train_x.head()
#train_x['Sex'] = pd.Categorical(train_x.Sex)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
train_x['Sex'] = label_encoder.fit_transform(train_x['Sex'])
test_x['Sex'] = label_encoder.fit_transform(test_x['Sex'])
train_x["Sex"].replace(np.NaN, 0)
test_x["Sex"].replace(np.NaN, 0)
train_x.info()
train_x["Age"] = train_x["Age"].fillna(0)
train_x["Age"] = train_x["Age"].astype(np.float32)
test_x["Age"] = test_x["Age"].fillna(0)
test_x["Age"] = test_x["Age"].astype(np.float32)
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_model.fit(train_x, train_y)
#rf_model.predict(train_x)
rf_model.score(train_x, train_y)
pred_y = rf_model.predict(test_x).round().astype(int)
out = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
out['Survived'] = pred_y
out.to_csv("../working/submission.csv", index = False)