import numpy as np # linear algebra

from scipy import stats

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers.core import Dense, Activation

from keras.models import Sequential

from keras import optimizers

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



print ('import completed')
model_1 = Sequential()

model_1.add(Dense(3, input_shape = (1,),  activation = 'tanh'))

#model_1.add(Dense(3, input_shape = (1,),  activation = 'relu'))

model_1.add(Dense(1))

model_1.compile(optimizer = 'sgd' , loss = 'mean_squared_error')

print(model_1.summary())
# A feature is an input variable, in this case x

x=np.linspace(start=-1, stop=1, num=50)

# A label is the thing we're predicting, in this case y

y=np.square(x)

plt.plot(x,y)
# train the model

history = model_1.fit(x, y, epochs = 20000, verbose = 0)

# Predict the values

y_pred = model_1.predict(x)

# Plot the predicted values (red) together with the actual values (blue)

plt.plot(x,y_pred, 'r',x,y,'b')
# A feature is an input variable, in this case x

x=np.linspace(start=-1, stop=1, num=50)

# A label is the thing we're predicting, in this case y

y=np.sin(x*3)

plt.plot(x,y)
model_3 = Sequential()

model_3.add(Dense(3, input_shape = (1,),  activation = 'tanh'))

#model_3.add(Dense(3, input_shape = (1,),  activation = 'relu'))

model_3.add(Dense(1))

model_3.compile(optimizer = 'sgd' , loss = 'mean_squared_error')

# train the model

history = model_3.fit(x, y, epochs = 20000, verbose = 0)

# Predict the values

y_pred = model_3.predict(x)

# Plot the predicted values (red) together with the actual values (blue)

plt.plot(x,y_pred, 'r',x,y,'b')
# A feature is an input variable, in this case x

x=np.linspace(start=-1, stop=1, num=50)

# A label is the thing we're predicting, in this case y

y=np.abs(x)

plt.plot(x,y)
model_2 = Sequential()

model_2.add(Dense(3, input_shape = (1,),  activation = 'tanh'))

#model_2.add(Dense(3, input_shape = (1,),  activation = 'relu'))

model_2.add(Dense(1))

model_2.compile(optimizer = 'sgd' , loss = 'mean_squared_error')

# train the model

history = model_2.fit(x, y, epochs = 20000, verbose = 0)

# Predict the values

y_pred = model_2.predict(x)

# Plot the predicted values (red) together with the actual values (blue)

plt.plot(x,y_pred, 'r',x,y,'b')
# A feature is an input variable, in this case x

x=np.linspace(start=-1, stop=1, num=50)

y=np.heaviside(x,0)

plt.plot(x,y)
model_4 = Sequential()

model_4.add(Dense(3, input_shape = (1,),  activation = 'tanh'))

#model_4.add(Dense(3, input_shape = (1,),  activation = 'relu'))

model_4.add(Dense(1))

model_4.compile(optimizer = 'sgd' , loss = 'mean_squared_error')

# train the model

history = model_4.fit(x, y, epochs = 20000, verbose = 0)

# Predict the values

y_pred = model_4.predict(x)

# Plot the predicted values (red) together with the actual values (blue)

plt.plot(x,y_pred, 'r',x,y,'b')
history = model_4.fit(x, y, epochs = 200000, verbose = 0)

# Predict the values

y_pred = model_4.predict(x)

# Plot the predicted values (red) together with the actual values (blue)

plt.plot(x,y_pred, 'r',x,y,'b')
# A feature is an input variable, in this case x

x=np.linspace(start=-1, stop=1, num=50)

# A label is the thing we're predicting, in this case y

y=stats.norm.pdf(x, loc=0, scale=1.5) #loc = mean; scale = standard deviation

plt.plot(x,y)
model_5 = Sequential()

model_5.add(Dense(3, input_shape = (1,),  activation = 'tanh'))

#model_5.add(Dense(3, input_shape = (1,),  activation = 'relu'))

model_5.add(Dense(1))

model_5.compile(optimizer = 'sgd' , loss = 'mean_squared_error')

# train the model

history = model_5.fit(x, y, epochs = 200000, verbose = 0)

# Predict the values

y_pred = model_5.predict(x)

# Plot the predicted values (red) together with the actual values (blue)

plt.plot(x,y_pred, 'r',x,y,'b')