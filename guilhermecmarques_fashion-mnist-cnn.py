%matplotlib inline

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
pure = np.load('../input/train_images_pure.npy')
f, ax = plt.subplots(2,5)
f.set_size_inches(80, 40)
for i in range(10):
    if i < 5:
        ax[0][i].imshow(pure[i].reshape(28, 28))
    else:
        ax[1][i-5].imshow(pure[i].reshape(28, 28)) 
plt.show()
noisy = np.load('../input/train_images_noisy.npy')
f, ax = plt.subplots(2,5)
f.set_size_inches(80, 40)
for i in range(10):
    if i < 5:
        ax[0][i].imshow(noisy[i].reshape(28, 28))
    else:
        ax[1][i-5].imshow(noisy[i].reshape(28, 28))    
    
plt.show()
rotated = np.load('../input/train_images_rotated.npy')
f, ax = plt.subplots(2,5)
f.set_size_inches(80, 40)
for i in range(10):
    if i < 5:
        ax[0][i].imshow(rotated[i].reshape(28, 28))
    else:
        ax[1][i-5].imshow(rotated[i].reshape(28, 28))    
    
plt.show()
both = np.load('../input/train_images_both.npy')
f, ax = plt.subplots(2,5)
f.set_size_inches(80, 40)
for i in range(10):
    if i < 5:
        ax[0][i].imshow(both[i].reshape(28, 28))
    else:
        ax[1][i-5].imshow(both[i].reshape(28, 28))    
    
plt.show()
from sklearn.model_selection import train_test_split
train_y = pd.read_csv("../input/train_labels.csv")["label"]

# normalizo os valores do pixels, para valores entre 0 e 1 
train_x = rotated / 255

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

train_x = train_x.reshape(train_x.shape[0], 1, 28, 28).astype('float32')
val_x = val_x.reshape(val_x.shape[0], 1, 28, 28).astype('float32')
train_y_b = pd.read_csv("../input/train_labels.csv")["label"]

# normalizo os valores do pixels, para valores entre 0 e 1 
train_x_b = both / 255

train_x_b, val_x_b, train_y_b, val_y_b = train_test_split(train_x_b, train_y_b, test_size=0.2)

#faço a conversão da matriz contendo cada imagem para um vetor
train_x_b = train_x_b.reshape(train_x_b.shape[0], 1, 28, 28).astype('float32')
val_x_b = val_x_b.reshape(val_x_b.shape[0], 1, 28, 28).astype('float32')
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras import backend as K
K.set_image_dim_ordering('th')
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(1, 28, 28)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_b = model
model.summary()
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3)]
model.fit(train_x, train_y, validation_data=(val_x, val_y), 
          epochs=5, batch_size=200, 
          verbose=1, callbacks=callbacks)
scores = model.evaluate(val_x, val_y, verbose=0)
print("Acurácia - CNN simples - rotated: %.2f%%" % (scores[1]*100))
model_b.fit(train_x_b, train_y_b, validation_data=(val_x_b, val_y_b), 
          epochs=5, batch_size=200, 
          verbose=1, callbacks=callbacks)
scores_b = model.evaluate(val_x_b, val_y_b, verbose=0)
print("Acurácia - CNN simples - both: %.2f%%" % (scores_b[1]*100))
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 256
num_classes = 10
epochs = 50

#input image dimensions
img_rows, img_cols = 28, 28
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(1, 28, 28)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model_b = model
model.summary()
model.fit(train_x, train_y, validation_data=(val_x, val_y), 
          epochs=20, batch_size=200, 
          verbose=1, callbacks=callbacks)
scores = model.evaluate(val_x, val_y, verbose=0)
print("Acurácia - CNN pool - rotated: %.2f%%" % (scores[1]*100))
model_b.fit(train_x_b, train_y_b, validation_data=(val_x_b, val_y_b), 
          epochs=20, batch_size=200, 
          verbose=1)
scores_b = model_b.evaluate(val_x_b, val_y_b, verbose=0)
print("Acurácia - CNN pool - both: %.2f%%" % (scores_b[1]*100))
x_test = np.load('../input/Test_images.npy')
f, ax = plt.subplots(2,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[0][i].imshow(x_test[i].reshape(28, 28))
    ax[1][i].imshow(x_test[-i-1].reshape(28, 28))    
    
plt.show()
x_test = x_test / 255

x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
y_prob = model_b.predict(x_test)
y_classes = y_prob.argmax(axis=-1)
prediction = pd.DataFrame(y_classes, columns = ['label'])
prediction
prediction.to_csv("prediction.csv", index=True)
