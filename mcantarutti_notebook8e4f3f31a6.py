# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")
test.shape
train = pd.read_csv("../input/train.csv")
train.shape
train_s=train.iloc[:,1:80]
train_s.head()
train_s.shape
test_s=test.iloc[:,1:80]
test_s.head()
test_s.shape
train_t = train.iloc[:,80]
train_t.head()
train_s.dtypes
train_s.dtypes == np.object
columns_to_transform = train_s.dtypes.index[train_s.dtypes== np.object].tolist()
columns_to_transform
train_with_dummies = pd.get_dummies(train_s,columns=columns_to_transform)
test_with_dummies = pd.get_dummies(test_s,columns=columns_to_transform)
train_with_dummies.dtypes
np.isnan(train_with_dummies).any()
train_with_dummies=train_with_dummies.fillna(train_with_dummies.mean())
test_with_dummies=test_with_dummies.fillna(test_with_dummies.mean())
train_with_dummies.shape
test_with_dummies.shape
train_with_dummies.dtypes.index.tolist()
test_with_dummies=test_with_dummies.ix[:,train_with_dummies.dtypes.index.tolist()]
test_with_dummies.shape
test_with_dummies=test_with_dummies.fillna(0)
np.isnan(test_with_dummies).any()
reg = linear_model.LinearRegression()
reg.fit (train_with_dummies,train_t)
reg.coef_
reg.score(train_with_dummies,train_t)
test_p=reg.predict(test_with_dummies)
test_id = test.iloc[:,0]
test_id=pd.DataFrame(test_id,columns=['Id'])
test_id.head()
test_p=pd.DataFrame(test_p,columns=['SalePrice'])
test_p.head()
submit=pd.concat([test_id,test_p],axis=1)
submit.to_csv('max_submit.csv',header=True,index=False)
import matplotlib.pyplot as plt
train_with_dummies[:,1:20].hist()
plt.show()

# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(tran_s.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

