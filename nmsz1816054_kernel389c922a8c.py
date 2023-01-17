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

# Loading train data and test data into Pandas.
train = pd.read_csv("../input/train.csv", header=0)
test = pd.read_csv("../input/test.csv", header=0)
from sklearn import ensemble
#设定随机森林分类模型
rf=ensemble.RandomForestClassifier(100) #设定包含100个决策树

feature = [col for col in train.columns if col not in ['Cover_Type','Id']]
X_test = test[feature]
X_train= train[feature]
rf.fit(X_train, train['Cover_Type'])
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": rf.predict(X_test)})
sub.to_csv("etc.csv", index=False)