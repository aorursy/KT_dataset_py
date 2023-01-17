import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.utils import np_utils

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head(10)
X_train = train.drop("label",axis=True)

Y_train = train["label"]
X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
#Encoding the output class label (One-Hot Encoding)

Y_train = to_categorical(Y_train, num_classes = 10)

Y_train[2]
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("trainX shape:", X_train.shape, "\ntrainY shape:", Y_train.shape)
# train["label"].hist(bins=10)

classes = [0,1,2,3,4,5,6,7,8,9]

ax=sns.countplot(x="label", data=train)

ax.set_xticklabels(classes)

plt.xlabel("Numbers")

plt.title("Frequency of digits(0-9) in the data \n")

plt.show()
plt.figure(figsize=(10,5))

for i in range(10):

    

    plt.subplot(2,5,i+1)

    plt.imshow(np.array(train.iloc[:,1:][train["label"]==i].iloc[0,:]).reshape(28,28))

    plt.xticks([])

    plt.yticks([])

    

plt.suptitle("Visualising Numbers 0-9")    

plt.tight_layout()

plt.show()
datagen = ImageDataGenerator(

        featurewise_center=False,             # set input mean to 0 over the dataset

        samplewise_center=False,              # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,   # divide each input by its std

        zca_whitening=False,                  # apply ZCA whitening

        rotation_range=10,                    # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1,                     # Randomly zoom image 

        width_shift_range=0.1,                # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,               # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,                # randomly flip images

        vertical_flip=False)                  # randomly flip images





datagen.fit(X_train)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
model.compile(loss='categorical_crossentropy',

             optimizer='rmsprop',

             metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86),

                              epochs = 30, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // 86

                              , callbacks=[learning_rate_reduction])
predictions = model.predict(test, verbose=0)

predictions[0:5]
pred=[]

for i in list(range(0,len(predictions))):

    pred.append(np.argmax(predictions[i]))
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": pred})

submissions.to_csv("DR.csv", index=False, header=True)
# Plot the loss and accuracy curves for training and validation 

plt.figure(figsize=(10,10))

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)