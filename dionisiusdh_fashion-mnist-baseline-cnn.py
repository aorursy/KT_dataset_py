import numpy as np

import pandas as pd 
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

print(train.shape)

train.head()
test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

print(train.shape)

train.head()

train_arr = np.array(train, dtype='float32')

test_arr = np.array(test, dtype='float32')
def preprocess(arr):

    x = arr[:, 1:]/255.0

    y = arr[:, 0]

    

    return x, y
X_train, y_train = preprocess(train_arr)
X_test, y_test = preprocess(test_arr)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=36)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import utils



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation, BatchNormalization

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers



from tensorflow.keras.preprocessing.image import ImageDataGenerator
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
datagen = ImageDataGenerator(featurewise_center=False,

                            samplewise_center=False,

                            featurewise_std_normalization=False,

                            samplewise_std_normalization=False,

                            zca_whitening=False,

                            rotation_range=15,

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            horizontal_flip=True,

                            vertical_flip=False)



datagen.fit(X_train)
model = Sequential()



# First layer

model.add(Conv2D(128, kernel_size=(3,3), input_shape=(28,28,1), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))



# Second layer

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))



# Third layer

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))



# Fully Connected layers

model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

    

model.add(Dense(10, activation='softmax'))

    

model.summary()
model.compile(loss='sparse_categorical_crossentropy',

             optimizer=Adam(lr=0.0003),

             metrics=['accuracy'])
history = model.fit(datagen.flow(X_train, y_train, batch_size = 64),

                    steps_per_epoch = len(X_train) // 64, 

                    epochs = 5, 

                    validation_data= (X_valid, y_valid),

                    verbose=1)
# Save the model

model.save('my_model.h5')
scores = model.evaluate(X_test, y_test)
pd.DataFrame(history.history).plot()
pred = model.predict(X_test)
class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



y_pred = np.argmax(pred, axis=1)
from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
fig, axes = plt.subplots(5, 5, figsize=(12,12))

axes = axes.ravel()



for i in np.arange(25):

    axes[i].imshow(X_test[i].reshape(28,28))

    axes[i].set_title('True: %s \nPredict: %s' % (class_names[int(y_test[i])], class_names[y_pred[i]]))

    axes[i].axis('off')

    plt.subplots_adjust(wspace=1)
fig, axes = plt.subplots(5, 5, figsize=(12,12))

axes = axes.ravel()



miss_pred = np.where(y_pred != y_test)[0]

for i in np.arange(25):

    axes[i].imshow(X_test[miss_pred[i]].reshape(28,28))

    axes[i].set_title('True: %s \nPredict: %s' % (class_names[int(y_test[miss_pred[i]])],

                                                 class_names[y_pred[miss_pred[i]]]))

    axes[i].axis('off')

    plt.subplots_adjust(wspace=1)