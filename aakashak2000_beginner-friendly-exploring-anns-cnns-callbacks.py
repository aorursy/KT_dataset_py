import os
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.activations import relu, softmax
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import glorot_normal
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization,Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Activation
sns.set()
files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        files.append(os.path.join(dirname, filename))
train_data = pd.read_csv(files[2])
test_data = pd.read_csv(files[0])
submission = pd.read_csv(files[1])
#checking number of training and testing examples in hand.
train_data.shape, test_data.shape
train_samples = train_data.drop(['label'], 1)
train_labels = train_data.label
#Plotting 50 images from the training set
fig,ax = plt.subplots(10, 5, figsize=(20,20))
axes = ax.ravel()
for i in range(50):
    image = np.array(train_samples.iloc[i+16,:]).reshape(28,28)
    axes[i].imshow(image)
#Plotting 50 images from the test set
fig,ax = plt.subplots(10, 5, figsize=(20,20))
axes = ax.ravel()
for i in range(50):
    image = np.array(test_data.iloc[i+16,:]).reshape(28,28)
    axes[i].imshow(image)
#Visualizing number of samples per class in the training set
sns.countplot(train_labels)
#Normalizing Pixel Values to a range of [0,1]
train_samples_scaled = train_samples.astype(float)/255
test_samples_scaled = test_data.astype(float)/255
#One-Hot Encoding target variables in the train data
encoded_train_labels = to_categorical(train_labels)
X_train = train_samples_scaled
y_train = encoded_train_labels

X_test = test_samples_scaled
model = Sequential()

model.add(Dense(128, activation = 'relu', input_shape = (784,)))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(32, activation='relu'))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, activation = 'softmax'))

opt = Adam(0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_split=0.05, epochs = 40, batch_size = 64)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('accuracy/loss')
plt.ylabel('epochs')
plt.legend()

es = EarlyStopping(monitor='val_loss', mode = 'min', patience = 20)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)


model = Sequential()

model.add(Dense(128, activation='relu', input_shape = (784,)))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64,activation='relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(32,activation='relu'))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, activation = 'softmax'))

opt = Adam(0.01)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_split=0.05, epochs = 40, batch_size = 64, callbacks = [es, rlrop])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('accuracy/loss')
plt.ylabel('epochs')
plt.legend()
model.save('ann_model.hdf5')
predictions = model.predict_classes(X_test)
submission.Label = predictions
submission.to_csv('/kaggle/working/ANN_submission.csv', index=False)
cnn_train = np.array(train_samples_scaled).reshape(-1,28,28,1)
cnn_test = np.array(test_samples_scaled).reshape(-1,28,28,1)
es = EarlyStopping(monitor='val_loss', mode = 'min', patience = 12)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001)

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides = ((2,2))))

model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides = (2,2)))
model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4096,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(cnn_train, y_train, validation_split=0.01, epochs = 50, batch_size=256, callbacks = [es, rlrop])
fig, ax = plt.subplots(1,2,figsize = (20,10))

ax[0].plot(history.history['accuracy'], label = 'Training Accuracy', color='r')
ax[0].plot(history.history['val_accuracy'],label = 'Validation Accuracy', color='b')
ax[0].legend(loc='lower right')

ax[1].plot(history.history['loss'], label = 'Training Loss', color='r')
ax[1].plot(history.history['val_loss'], label = 'Validation Loss', color = 'b')
ax[1].legend(loc='upper right')

es = EarlyStopping(monitor='val_loss', mode = 'min', patience = 12, restore_best_weights=True)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0000001)

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides = ((2,2))))

model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2), strides = (2,2)))
model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(cnn_train, y_train, validation_split=0.01, epochs = 50, batch_size=256, callbacks = [es, rlrop])
fig, ax = plt.subplots(1,2,figsize = (20,10))

ax[0].plot(history.history['accuracy'], label = 'Training Accuracy', color='r')
ax[0].plot(history.history['val_accuracy'],label = 'Validation Accuracy', color='b')
ax[0].legend(loc='lower right')

ax[1].plot(history.history['loss'], label = 'Training Loss', color='r')
ax[1].plot(history.history['val_loss'], label = 'Validation Loss', color = 'b')
ax[1].legend(loc='upper right')
model.save('cnn_model.hdf5')
predictions = model.predict_classes(cnn_test)
submission.Label = predictions
submission.to_csv('/kaggle/working/CNN_submission.csv', index=False)