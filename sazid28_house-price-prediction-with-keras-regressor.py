import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import Sequential
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("../input/train.csv",delimiter=",")
data.head()
data.tail()
data.columns
X = data[['ID', 'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
       'tax', 'ptratio', 'black', 'lstat']]
X
Y = data[['medv']]
Y
def baseline_model():
# create model
    model = Sequential()
    model.add(Dense(16, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
model.compile(loss='mean_squared_error', optimizer='adam')
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
