import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
train.head()
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1']].fillna(0.0) , train['SalePrice'])
y_pred = lr.predict(test[['LotArea', 'OverallQual', 'OverallCond', 'BsmtFinSF1']].fillna(0.0))
sub['SalePrice'] = y_pred
sub.to_csv('my_submission.csv', index=False)