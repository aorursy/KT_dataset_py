import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

"""for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))"""
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
dataset = pd.read_csv('../input/watercons/water-consumption-in-the-new-york-city.csv')
print(dataset)
x=dataset.iloc[:,0:2].values

y=dataset.iloc[:,2].values

y=np.reshape(y, (-1,1))

scaler_x = MinMaxScaler()

scaler_y = MinMaxScaler()

print(scaler_x.fit(x))

xscale=scaler_x.transform(x)

print(scaler_y.fit(y))

yscale=scaler_y.transform(y)
plt.plot(xscale,yscale)

plt.title('Water Consumption')

plt.ylabel('Dependent variable (y)')

plt.xlabel('Independent variable (x)')

plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(xscale, yscale, test_size = 0.1, random_state = 0)
import keras 

from keras import backend as k

from keras.models import Sequential

from keras.layers import Dense

from keras.layers.core import Activation

from keras.optimizers import Adam



"""classifier.add(Dense(output_dim=1,init= "uniform", activation = 'relu',input_dim = 1))# Input layer

classifier.add(Dense(output_dim=1,init= "uniform",activation = 'relu')) # Hidden layer

classifier.add(Dense(output_dim =1,init= "uniform",activation = 'softmax'))

classifier.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])

classifier.fit(x_train,y_train, batch_size = 2,epochs =10)"""

#model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

from keras.layers import LeakyReLU

model = Sequential()

model.add(Dense(32, input_dim=2, activation ='relu' ))

model.add(Dense(16, activation='relu' ))

model.add(Dense(12, activation='relu' ))

model.add(Dense(1, activation='linear' ))

#model.add(Dense(1))

#model.add(LeakyReLU(alpha=0.05))

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

#model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=164, batch_size=64)
Xnew = np.array([[2007,8125076.5]])

Xnew= scaler_x.transform(Xnew)

ynew= model.predict(Xnew)

print(ynew)

print(Xnew)

#invert normalize

ynew = scaler_y.inverse_transform(ynew) 

Xnew = scaler_x.inverse_transform(Xnew)

print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

y_pred = model.predict(x_test)

plt.plot(y_test, color = 'red', label = 'Real data')

plt.plot(y_pred, color = 'blue', label = 'Predicted data')

plt.title('Prediction')

plt.legend()

plt.show()
print(history.history.keys())

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()




