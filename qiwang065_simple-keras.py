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
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
X = [[2.1017802576,2.0941102339,1.7880408632,1.1675915421],

[1.8240966796,1.7999390543,1.5393686454,1.2158788002],

[2.8306066915,2.5751854860,2.3012473627,2.3244383211],

[2.0526577830,2.0402799354,1.7517159653,1.1828612175],

[1.8029634579,1.7776919005,1.5250027704,1.1834025237],

[2.8176283880,2.5580875400,2.2954364023,2.3471228709],

[2.0196752126,2.0016048634,1.7268363853,1.2083807500],

[1.7945214383,1.7674151935,1.5197459097,1.1297774311],

[2.8951436678,2.6188457359,2.3742241195,2.4568020124]]

Y =  [[0.75,3.75],

[0.75,4.5],

[0.75,5.25],

[1.25,3.75],

[1.25,4.5],

[1.25,5.25],

[1.75,3.75],

[1.75,4.5],

[1.75,5.25]]
X_train = pd.DataFrame(X)

y_train = pd.DataFrame(Y)
def build_regressor():

    regressor = Sequential()

    regressor.add(Dense(units=3, input_dim=4))

    regressor.add(Dense(units=2))

    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae'])

    return regressor
from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=9,epochs=100000)
results=regressor.fit(X_train,y_train)
y_pred= regressor.predict(X_train)

y_pred,y_train