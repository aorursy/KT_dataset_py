import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

dataset = pd.read_csv("../input/house.csv")

dataset.head(2)

dataset.describe(include='all')

sns.pairplot(dataset)

X=dataset.iloc[:,0:11]

y=dataset.iloc[:,11].values

from sklearn.preprocessing import  MinMaxScaler

sc= MinMaxScaler()

X= sc.fit_transform(X)

y= y.reshape(-1,1)

y=sc.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



from keras import Sequential

from keras.layers import Dense

def build_regressor():

    regressor = Sequential()

    regressor.add(Dense(units=100, input_dim=11))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])

    return regressor

from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)

results=regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
