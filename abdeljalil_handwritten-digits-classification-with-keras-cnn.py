import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sn

%matplotlib inline

palette = sn.color_palette('Set3')

sn.set(style='white', context='notebook')
srandom = 42
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(3)
X_train = train.drop(['label'], axis=1)

Y_train = train['label']



sn.countplot(Y_train, palette=palette);
print('There are %d missing values.' % X_train.isna().sum().sum())
X_train = X_train.values.reshape(-1,28, 28,1)

test = test.values.reshape(-1,28, 28,1)
fig, ax = plt.subplots(4,4,figsize=(6, 6))

for i in range(4):

    for j in range(4):

        ax[i,j].imshow(X_train[np.random.randint(len(X_train))][:,:,0], cmap='gray_r')

        ax[i,j].set_axis_off()
X_train = X_train.astype('float32')

test = test.astype('float32')



X_train = X_train / 255.0

test = test / 255.0
from keras.utils.np_utils import to_categorical;
Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size= 0.1, random_state=srandom)
from keras.layers import Conv2D, MaxPool2D,Flatten, Dense, Dropout

from keras.models import Sequential
def create_model():

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))

    model.add(Dropout(0.25))

    model.add(Dense(10, activation = 'softmax'))

    return model
model = create_model()
from keras.optimizers import RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, Y_train, epochs=30, batch_size=63, validation_data = (X_val,Y_val), verbose=2)
print('History object contains: %s' % history.history.keys())
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

ax1.plot(history.history['acc'])

ax1.plot(history.history['val_acc'])

ax1.legend(['train', 'validation'], loc='best')

ax1.set(xlabel='Epoch', ylabel='Accuracy')

ax1.set_title('Model accuracy');



ax2.plot(history.history['loss'])

ax2.plot(history.history['val_loss'])

ax2.legend(['train', 'validation'], loc='best')

ax2.set(xlabel='Epoch', ylabel='Loss')

ax2.set_title('Model loss');



Y_predict = model.predict(test)
plt.imshow(test[0][:,:,0], cmap='gray_r');
Y_predict[0]
print('The first sample in the test set is: %d' % Y_predict[0].argmax())
Y_predict_class = np.argmax(Y_predict, axis=1)
np.arange(0,len(Y_predict_class)).shape
predictions = pd.DataFrame({'ImageId' : np.arange(1,len(Y_predict_class)+1), 'Label' : Y_predict_class})
predictions.to_csv('mnist_cnn_keras.csv', index=False)