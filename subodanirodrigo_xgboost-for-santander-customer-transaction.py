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
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
#Check the shape

print(train.shape)

print(test.shape)
#train dataset

train.head()
test.head()
#all the Predicter variables are continuous

train.describe()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    a = pd.concat([total], axis=1, keys=['Total'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    a['Types'] = types

    return(np.transpose(a))
%%time

missing_data(train)
%%time

missing_data(test)
#Check target(response variable)

train['target'].value_counts().plot.bar();
Y = train.values[:,1]

x=train.values[:,2:203]

import xgboost as xgb

from xgboost.sklearn import XGBClassifier
model = XGBClassifier(learning_rate =0.1,n_estimators=100,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)

#fit the model for training dataset

model.fit(x,Y)
#select test features from test data

x_test=test.values[:,1:202]
#make predictions for test set based on training

y_pred = model.predict(x_test)

print(y_pred)
#submit the submission

submission = pd.DataFrame({"ID_code": test.ID_code.values})

submission["target"] =y_pred

submission.to_csv("submission.csv", index=False)
submission.head()