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
import numpy as np #basic for math functions

from sklearn.datasets import load_iris #to load IRIS DataSet

from sklearn.model_selection import GridSearchCV #to import GridSearchCV lib

from sklearn.preprocessing import StandardScaler,OneHotEncoder #to manipulate Data

#Now we will import DL lib

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
def Create_Model():

    model = Sequential()

    model.add(Dense(12,input_shape=(4,),activation="relu"))

    model.add(Dense(3,activation="softmax"))

    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

    return model
iris = load_iris()

x = iris.data

y = iris.target



scaller = StandardScaler()

X = scaller.fit_transform(x)

Y = y.reshape(-1,1)



ohe = OneHotEncoder()

Y = ohe.fit_transform(Y).toarray()



model = KerasClassifier(build_fn=Create_Model, verbose=0)

batch_size=[20]

epochs=[10]

param_grid = dict(batch_size=batch_size,epochs=epochs)

grid = GridSearchCV(estimator=model,param_grid=param_grid,cv = 3,n_jobs=1)

grid_result = grid.fit(X,Y)



print("Best:{} using {}".format(grid_result.best_score_,grid_result.best_params_))
def Create_model(optimizer = "adam"):

    model = Sequential()

    model.add(Dense(12, input_shape = (4,), activation = "relu"))

    model.add(Dense(3, activation = "softmax"))

    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

    return model



np.random.seed(12)

### Data IRIS

iris = load_iris()

x = iris.data

y = iris.target

scaller = StandardScaler()

X = scaller.fit_transform(x)

Y = y.reshape(-1, 1)

ohe = OneHotEncoder()

Y = ohe.fit_transform(Y).toarray()

model = KerasClassifier(build_fn=Create_model, verbose = 0)

optimizer = ["sgd", "adam", "RMSProp", "Adagrad", "Adamax"]

param_grid = dict(optimizer = optimizer)

grid = GridSearchCV(estimator=model, param_grid= param_grid, cv = 3, n_jobs=1)

grid_result = grid.fit(X, Y)

print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))
def Create_model(activation = "relu"):

    model = Sequential()

    model.add(Dense(12, input_shape = (4,), activation = activation))

    model.add(Dense(3, activation = activation))

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model



np.random.seed(12)

### Data IRIS

iris = load_iris()

x = iris.data

y = iris.target

scaller = StandardScaler()

X = scaller.fit_transform(x)

Y = y.reshape(-1, 1)

ohe = OneHotEncoder()

Y = ohe.fit_transform(Y).toarray()

model = KerasClassifier(build_fn=Create_model, verbose = 0)

activation = ["softmax","relu","tanh","sigmoid"]

param_grid = dict(activation = activation)

grid = GridSearchCV(estimator=model, param_grid= param_grid, cv = 3, n_jobs=1)

grid_result = grid.fit(X, Y)

print("Best: {} using {}".format(grid_result.best_score_,grid_result.best_params_))