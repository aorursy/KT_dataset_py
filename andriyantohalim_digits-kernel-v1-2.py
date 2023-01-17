import numpy as np
import pandas as pd

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.describe()
df_test.describe()
y_train = df_train["label"]
y_train.describe()
X_train = df_train.drop(columns="label")
X_train.describe()
X_test = df_test
X_train = np.array((df_train.iloc[:,1:].values).astype('float32'))
X_train.min()
X_train.max()
X_test = np.array(df_test.values.astype('float32'))
X_train.min()
X_train.max()
X_train = X_train/255.0
X_train.max()
X_test = X_test/255.0
X_test.max()
y_train = np.array(df_train.iloc[:,0].values.astype('int32'))
y_train.min()
y_train.max()
X_train = X_train.reshape(-1, 28, 28, 1)
X_train.shape
X_test = X_test.reshape(-1, 28, 28, 1)
X_test.shape
plt.subplot(171)
plt.title(y_train[1])
plt.imshow(X_train[1][:,:,0])

plt.subplot(172)
plt.title(y_train[2])
plt.imshow(X_train[2][:,:,0])

plt.subplot(173)
plt.title(y_train[3])
plt.imshow(X_train[3][:,:,0])

plt.subplot(174)
plt.title(y_train[4])
plt.imshow(X_train[4][:,:,0])

plt.subplot(175)
plt.title(y_train[5])
plt.imshow(X_train[5][:,:,0])

plt.subplot(176)
plt.title(y_train[6])
plt.imshow(X_train[6][:,:,0])

plt.subplot(177)
plt.title(y_train[7])
plt.imshow(X_train[7][:,:,0])
plt.subplot(151)
plt.imshow(X_test[1][:,:,0])

plt.subplot(152)
plt.imshow(X_test[2][:,:,0])

plt.subplot(153)
plt.imshow(X_test[3][:,:,0])

plt.subplot(154)
plt.imshow(X_test[4][:,:,0])

plt.subplot(155)
plt.imshow(X_test[5][:,:,0])
from keras.utils.np_utils import to_categorical

y_train_encoded = to_categorical(y_train, num_classes=10)
for idx in range (20, 30):
    print (y_train[idx], y_train_encoded[idx])
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train_encoded, test_size = 0.1, random_state=2)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense , Dropout , Lambda, Flatten

model = Sequential()

model.add(Conv2D(filters = 32, 
                 kernel_size = (5,5),
                 padding = 'Same', 
                 activation ='relu', 
                 input_shape = (28,28,1)))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters = 32, 
                 kernel_size = (5,5),
                 padding = 'Same', 
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, 
                 kernel_size = (3,3),
                 padding = 'Same', 
                 activation ='relu'))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters = 64, 
                 kernel_size = (3,3),
                 padding = 'Same', 
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
#model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
from keras.optimizers import Adam ,RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.compile(optimizer = optimizer , 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
batches = datagen.flow(X_train, Y_train, batch_size=96)
val_batches = datagen.flow(X_val, Y_val, batch_size=96)
history = model.fit_generator(generator=batches, 
                            epochs=1,
                            validation_data=val_batches,
                            validation_steps=val_batches.n,
                            verbose=2,
                            steps_per_epoch=batches.n,
                            callbacks=[learning_rate_reduction])
predictions = model.predict_classes(X_test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                          "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)
