# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20,5))
sns.heatmap(df.corr(),annot = True)
def is_renovated(x):
    if x == 0:
        return 0 
    else:
        return 1 
df['is_renovated'] = df.yr_renovated.apply(is_renovated)
df.drop(['id','date','yr_renovated'],axis = 1,inplace=True)

X = df.iloc[:,1:].values
y = df.iloc[:,0:1].values
y = y/1000
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
def base_model():
    model = Sequential()
    model.add(Dense(units=18,input_dim=18,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dense(units=7,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dense(units=1,kernel_initializer = 'normal',activation = 'relu'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model
kfold = KFold(n_splits=10)
estimator = KerasRegressor(build_fn=base_model, epochs=100, batch_size=25)

results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
