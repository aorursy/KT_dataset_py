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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sub = pd.read_csv('../input/sample_submission.csv')

#note their is no Survived column here which is our target varible we are trying to predict

train.info()

#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset

#(rows,columns)

train.shape

#Describe gives statistical information about numerical columns in the dataset

train.describe()
train_feat = train.drop(columns=['id', 'target'], axis=1)

test_feat = test.drop(columns='id', axis=1)

from sklearn.preprocessing import StandardScaler

std = StandardScaler()



train_feat = std.fit_transform(train_feat)

test_feat = std.fit_transform(test_feat)





train_label = train['target']



from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV, SGDClassifier



l1 = LassoCV().fit(train_feat, train_label)

l2 = LogisticRegressionCV().fit(train_feat, train_label)

pred3 = l1.predict(test_feat)  # 0.846

pred4 = l2.predict(test_feat)  # 0.6

target = pd.DataFrame(pred3).rename(columns={0: 'target'})

sub_id = sub[['id']]

submission = pd.concat([sub_id, target], axis=1)

print(submission)



submission.to_csv('sub.csv', index=False)