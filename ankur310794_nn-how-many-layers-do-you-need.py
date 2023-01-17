#loading Keras, numpy, and matplotlib.
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3107)
# constants
npts = 100 # points per blob
tot_npts = 4*npts # total number of points 
s = 0.005 # ~standard deviation
sigma = np.array([[s, 0], [0, s]]) #cov matrix
# Generate Data
data1 = np.random.multivariate_normal( [0,0], sigma, npts)
data2 = np.random.multivariate_normal( [0,1], sigma, npts)
data3 = np.random.multivariate_normal( [1,0], sigma, npts)
data4 = np.random.multivariate_normal( [1,1], sigma, npts)

and_data = np.concatenate((data1, data2, data3, data4)) # data
and_labels = np.concatenate((np.ones((3*npts)),np.zeros((npts)))) # labels
#print(and_data.shape)
#print(and_labels.shape)

plt.figure(figsize=(20,10))
plt.scatter(and_data[:,0][and_labels==0], and_data[:,1][and_labels==0],c='b')
plt.scatter(and_data[:,0][and_labels==1], and_data[:,1][and_labels==1],c='r')

plt.plot()
plt.title('AND problem',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid('on')
plt.show()
#(400, 2)
#(400,)
model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(and_data, 
                    and_labels, 
                    epochs=200,
                    batch_size=16,
                    verbose=0)
history_dict = history.history
history_dict.keys()
plt.figure(figsize=(20,10))
plt.subplot(121)
loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.subplot(122)
acc_values = history_dict['acc']
#val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# Generate Data
data1 = np.random.multivariate_normal( [0,0], sigma, npts)
data2 = np.random.multivariate_normal( [0,1], sigma, npts)
data3 = np.random.multivariate_normal( [1,0], sigma, npts)
data4 = np.random.multivariate_normal( [1,1], sigma, npts)

or_data = np.concatenate((data1, data2, data3, data4))
or_labels = np.concatenate((np.ones((npts)),np.zeros((3*npts))))

plt.figure(figsize=(20,10))
plt.scatter(or_data[:,0][or_labels==0], or_data[:,1][or_labels==0],c='b')
plt.scatter(or_data[:,0][or_labels==1], or_data[:,1][or_labels==1],c='r')
plt.title('OR problem',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid('on')
plt.show()
model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(or_data, 
                    or_labels, 
                    epochs=400,
                    batch_size=16,
                    verbose=0)


history_dict = history.history
history_dict.keys()


plt.figure(figsize=(20,10))
plt.subplot(121)
loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.subplot(122)
acc_values = history_dict['acc']
#val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
#plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# Generate Data
data1 = np.random.multivariate_normal( [0,0], sigma, npts)
data2 = np.random.multivariate_normal( [0,1], sigma, npts)
data3 = np.random.multivariate_normal( [1,0], sigma, npts)
data4 = np.random.multivariate_normal( [1,1], sigma, npts)

xor_data = np.concatenate((data1, data4, data2, data3))
xor_labels = np.concatenate((np.ones((2*npts)),np.zeros((2*npts))))

plt.figure(figsize=(20,10))
plt.scatter(xor_data[:,0][xor_labels==0], xor_data[:,1][xor_labels==0],c='b')
plt.scatter(xor_data[:,0][xor_labels==1], xor_data[:,1][xor_labels==1],c='r')
plt.title('XOR problem',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid('on')
plt.show()
model = models.Sequential()
model.add(layers.Dense(1, activation='sigmoid', input_shape=(2,)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_10 = history.history

model = models.Sequential()
model.add(layers.Dense(1, activation='relu', input_shape=(2,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_11 = history.history

model = models.Sequential()
model.add(layers.Dense(2, activation='relu', input_shape=(2,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_21 = history.history


history = model.fit(xor_data, 
                    xor_labels, 
                    epochs=400,
                    batch_size=32,
                    verbose=0)
history_dict_41 = history.history
plt.figure(figsize=(20,10))
plt.subplot(121)
loss_values = history_dict_10['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, history_dict_10['loss'], 'o', label='Training loss')
plt.plot(epochs, history_dict_11['loss'], 'o', label='Training loss')
plt.plot(epochs, history_dict_21['loss'], 'o', label='Training loss')
plt.title('Training loss',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.legend(['1_0','1_1','2_1','4_1'],fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


plt.subplot(122)
acc_values = history_dict_10['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, history_dict_10['acc'], 'o', label='Training loss')
plt.plot(epochs, history_dict_11['acc'], 'o', label='Training loss')
plt.plot(epochs, history_dict_21['acc'], 'o', label='Training loss')
plt.title('Training accuracy',fontsize=20)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(['1_0','1_1','2_1','4_1'],fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
