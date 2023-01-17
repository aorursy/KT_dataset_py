import numpy as np

import pandas as pd
red = pd.read_csv("../input/winequality-red.csv")
red.tail()
red.info()
#red['quality'] = red['quality'].astype('category')
red.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set(xlabel = 'total sulfur dioxide',

      ylabel = 'free sulfur dioxide')

ax.scatter(red['total sulfur dioxide'], red['free sulfur dioxide'], c='r')

plt.show()
#red['quality'].describe()
import seaborn as sns

import matplotlib.pyplot as plt



corr = red.corr()

sns.heatmap(corr, xticklabels = corr.columns.values,

           yticklabels=corr.columns.values)

red.columns
red['quality'].hist()
red['total sulfur dioxide'].hist()
red['fixed acidity'].hist()
red['volatile acidity'].hist()
red['citric acid'].hist()
red['residual sugar'].hist()
red['chlorides'].hist()
red['free sulfur dioxide'].hist()
from sklearn.model_selection import train_test_split



X = red.iloc[:,0:11]

y = np.ravel(red.quality)



X_train, X_test, y_train, y_test = train_test_split(X,

                                                  y,

                                                  test_size=0.33,

                                                  random_state = 42)
X.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(164, input_dim= 11, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(1))
model.compile(loss= 'mse',

             optimizer='rmsprop',

             metrics=['mse'])

model.fit(X_train, y_train, epochs=50, batch_size=1,verbose=1)
mse_value, mae_value = model.evaluate(X_test, y_test, verbose=0)



print(mse_value)
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score



r2_score(y_test, y_pred)