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
train = pd.read_csv("../input/comp_train.csv")

test = pd.read_csv("../input/comp_test.csv")

print(train.shape)
train.head()
test.head()
import matplotlib.pyplot as plt
plt.scatter(train["OverallQual"],train["SalePrice"])
plt.scatter(train["TotalBsmtSF"],train["SalePrice"])
plt.scatter(train["YearBuilt"],train["SalePrice"])
X_train = train["TotalBsmtSF"].values

y_train = train["SalePrice"].values





import numpy as np

m=train.shape[0]



#changing the shape to ,x1

one=np.ones((m,1))

X_train = X_train.reshape((m,1))

y_train = y_train.reshape((m,1))

X1=np.hstack((X_train,one))

theta=np.dot(np.linalg.pinv(X1),y_train)

print(theta)
import numpy as np

m=train.shape[0]

X_train=train.iloc[:,1:-1].values

y_train=train['SalePrice'].values.reshape((m,1))



one=np.ones((m,1))

X2=np.hstack((X_train,one))

theta=np.dot(np.linalg.pinv(X2),y_train)

print(theta)
X_test=test.iloc[:,1:].values

m_test=X_test.shape[0]

one=np.ones((m_test,1))

X_test=np.hstack((X_test,one))
prediction=np.dot(X_test,theta)

prediction
sub=pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice']=prediction

sub.to_csv("prediction.csv", index = False)
print(prediction.shape)