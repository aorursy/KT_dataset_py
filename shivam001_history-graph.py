# dependencies
import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# 7 seeds
seed = 7
np.random.seed(seed)
# dataset
dataset = np.loadtxt("../input/pimaindiansdiabetes.data.csv", delimiter=",")
# x -> features .  y-> labels
X = dataset[:,0:8]
print(X[:5])
Y = dataset[:,8]
print(Y[:5])
# Model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
# use sigmoid activation function bcz we are interested in 2 output in final layer
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# training
history = model.fit(X, Y, validation_split=0.30, epochs=200, batch_size=10)

print(history.history.keys())
# accuracy summary
plt.plot(history.history['acc'],color='y')
plt.plot(history.history['val_acc'],color='r')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# loss summary
plt.plot(history.history['loss'],color='b')
plt.plot(history.history['val_loss'],color='c')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
print("please upvote if you like kernel: thanks,happy learning")