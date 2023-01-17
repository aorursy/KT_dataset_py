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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.models import *

from keras.layers import *

from keras.losses import mean_squared_logarithmic_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression,LinearRegression
train = pd.read_csv('/kaggle/input/train.csv')

test = pd.read_csv('/kaggle/input/test.csv')
train.info()
test.info()
col=np.array(train.describe().columns)

data=train.loc[:,col]

data.info()
data=data.drop(columns=['Id','LotFrontage','GarageYrBlt','MasVnrArea'])

data.info()
X=data.values[:,:-1]

y = data.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rfR=RandomForestRegressor(max_depth=10,random_state=42,criterion='mae')
rfR.fit(X_train,y_train)
rfR.score(X_test,y_test)
dataTest=test.loc[:,col]

dataTest=dataTest.drop(columns=['Id','LotFrontage','GarageYrBlt','MasVnrArea'])

dataTest.info()
dataTest.fillna(0,inplace=True)
dataTest.info()
test_X=dataTest.values[:,:-1]

pred=rfR.predict(test_X)
pred
sample=pd.read_csv('/kaggle/input/sample_submission.csv')
sample.info()
sample.SalePrice=pred
sample.to_csv('/kaggle/output.csv',index=None)