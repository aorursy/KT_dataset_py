# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test.head()
X_train=train.drop(['label'],axis=1)

X_train.head()
Y_train=train['label']

Y_train
train.isnull().sum()

test.isnull().sum()
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier



regressor = KNeighborsClassifier()

regressor.fit(X_train, Y_train)
y_pred=regressor.predict(test)
y_pred
# prepare submit file



np.savetxt('submisson.csv', 

           np.c_[range(1,len(test)+1),y_pred], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')