import numpy as np
X = np.random.rand(20,1) * 10 -5
y = np.sin(X)
X.shape,y.shape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.backend as K
K.clear_session()
K.set_learning_phase(1)
model = Sequential()
model.add(Dense(1,input_dim = 1))


model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(20))
model.add(Activation('sigmoid'))
model.add(Dense(1))
from keras.optimizers import SGD
model.compile(loss='mse',optimizer=SGD(lr=0.01))
model.fit(X,y,epochs=10000,batch_size=10,verbose=0)
X_test = np.arange(-10,10,0.1)
X_test = np.expand_dims(X_test,-1)

probs = []
for i in range(100):
    out = model.predict(X_test)
    probs.append(out)
p = np.array(probs)
p.shape
mean = np.mean(p,axis=0)
std = np.std(p,axis=0)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.plot(X_test,mean,c='blue')

lower_bound = mean - std * 0.5
upper_bound =  mean + std * 0.5
plt.fill_between(X_test.flatten(),upper_bound.flatten(),lower_bound.flatten(),alpha=0.25, facecolor='blue')

lower_bound = mean - std
upper_bound =  mean + std
plt.fill_between(X_test.flatten(),upper_bound.flatten(),lower_bound.flatten(),alpha=0.25, facecolor='blue')

lower_bound = mean - std * 2
upper_bound =  mean + std * 2
plt.fill_between(X_test.flatten(),upper_bound.flatten(),lower_bound.flatten(),alpha=0.25, facecolor='blue')

plt.scatter(X,y,c='black')
