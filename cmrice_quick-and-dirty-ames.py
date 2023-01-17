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
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
train_url = '../input/train1.csv'
test_url = '../input/test1.csv'
train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

X = np.array(train['LotArea'])
y = np.array(train.SalePrice)
X = X.reshape(-1,1)
lr = LinearRegression()
lr.fit(X,y)
preds = lr.predict(X)
lr.score(X,preds)
test_feat = ['LotArea']
X_test = np.array(test[test_feat])
X_test = X_test.reshape(-1,1)
test_preds = lr.predict(X_test)

