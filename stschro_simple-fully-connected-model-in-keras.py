import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import preprocessing
import matplotlib.pyplot as plt
#import train data in Pandas DataFrame
df_train = pd.read_csv('../input/digit-recognizer/train.csv').astype(float)

#import test data in Pandas DataFrame
df_test = pd.read_csv('../input/digit-recognizer/test.csv').astype(float)

#DataFrame df_train: Show first 5 entries
df_train.head()
#Split DataFrame df_train in 2 parts: X (inputs) & y (labels)
#Split & Scale (X_train_flat)
X_train_flat = preprocessing.scale(df_train[df_train.columns[1:]])

#Scale (X_test_flat)
X_test_flat = preprocessing.scale(df_test)

#One-Hot Encoding labels
y_train = to_categorical(df_train[df_train.columns[0]])

#Show first 5 rows of training labels (before One-Hot Enconding)
pd.DataFrame(df_train[df_train.columns[0]]).head()
#Show first 5 rows of training labels (after One-Hot Enconding)
pd.DataFrame(y_train).head()
#Plot one example picture
#Convert flattend data in to 2D array for each image with size of 28 x 28
X_train_2d = X_train_flat.reshape(42000, 28,28)
#Plot image #100
plt.imshow(X_train_2d[100])
#Very simple fully-connected (dense) layer model
model = Sequential()
#First layer with 64 units expects input of 784 (28 x 28)
model.add(Dense(units=64, activation='relu', input_dim=784))
#Second layer with 32 units
model.add(Dense(units=32, activation='relu'))
#Output layer with 10 units (for '0' to '9') 
model.add(Dense(units=10, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Fit Model (15 epochs, batch-size 64 and validation split of 30%)
history = model.fit(x=X_train_flat, y=y_train, batch_size=64, epochs=15, validation_split=0.3)
#Max validation accuracy during training
np.max(history.history['val_acc'])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training')
plt.plot(epochs, val_acc, 'b', label = 'Validierung')
plt.title('Correct Classification Rate training/validation')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Loss training')
plt.plot(epochs, val_loss, 'b', label='Loss Validation')
plt.title('Value of Loss Function training/validation')
plt.legend()

from keras.layers import Dropout

#Very simple fully-connected (dense) layer model
model = Sequential()
#add dropout
model.add(Dropout(0.2, input_shape=(784,)))
#First layer with 64 units expects input of 784 (28 x 28)
model.add(Dense(units=64, activation='relu'))
#add dropout
model.add(Dropout(0.2))
#Second layer with 32 units
model.add(Dense(units=32, activation='relu'))
#Output layer with 10 units (for '0' to '9') 
model.add(Dense(units=10, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Fit Model (15 epochs, batch-size 64 and validation split of 30%)
history = model.fit(x=X_train_flat, y=y_train, batch_size=64, epochs=25, validation_split=0.3)
#Max validation accuracy during training
np.max(history.history['val_acc'])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training')
plt.plot(epochs, val_acc, 'b', label = 'Validierung')
plt.title('Correct Classification Rate training/validation')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Loss training')
plt.plot(epochs, val_loss, 'b', label='Loss Validation')
plt.title('Value of Loss Function training/validation')
plt.legend()
#use trained model predict label (one hot encoded) for test data
y_hat_one_hot = model.predict(X_test_flat)

#convert one-hot encoded values into label values
y_hat = np.argmax(y_hat_one_hot, axis=1)
#write prediction into Pandas DataFrame
y_hat = pd.DataFrame(y_hat, columns=['Label'])
y_hat.index += 1 
y_hat.index.name = 'ImageId'

#Show first 5 rows of prediction table
y_hat.head()
