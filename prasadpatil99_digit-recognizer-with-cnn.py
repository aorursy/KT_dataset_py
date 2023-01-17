import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline 



import seaborn as sns
training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
training_data.head()
testing_data.head()
label = training_data.label.value_counts()

label
sns.countplot(training_data['label'])

plt.title("Distribution of feature")

plt.ylabel("Number of Occurences")

plt.xlabel("Labels")
train_data = training_data.drop(['label'], axis=1).values.astype('float32') # all pixel values

target = training_data['label'].values.astype('int32') # only labels i.e targets digits

test_data = testing_data.values.astype('float32')

train_data = train_data.reshape(train_data.shape[0], 28, 28) / 255.0

test_data = test_data.reshape(test_data.shape[0], 28, 28) / 255.0
import keras

num = 10

target = keras.utils.to_categorical(target,num)
plt.figure(figsize=(10,10))

for i in range(50):  

    plt.subplot(5, 10, i+1)

    plt.imshow(train_data[i].reshape((28,28)),interpolation='nearest')

plt.show()
import tensorflow.keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

num_classes = 10

input_shape = (28, 28, 1)
X_train, X_val, Y_train, Y_val = train_test_split(train_data, target , test_size = 0.1, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)  

X_val = X_val.reshape(X_val.shape[0], 28, 28,1)  

test_data = test_data.reshape(test_data.shape[0], 28, 28,1)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10, # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally 

        height_shift_range=0.1,  # randomly shift images vertically 

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)  # fitting X_train model 
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',   # quality to be monitored 

                                            patience=3,          # no of epoch with no improvement after learning rate will be reduced

                                            verbose=1,           # update message

                                            factor=0.5,          # reducing learning rate 

                                            min_lr=0.0001)       # lower bound learning rate 
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='Adam',

              metrics=['accuracy'])
fitting = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 90),

                              epochs = 50, validation_data = (X_val, Y_val),

                              verbose = 1, callbacks = [learning_rate_reduction])
print(model.summary())
predicted_classes = model.predict_classes(test_data)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("submission.csv", index=False, header=True)
submissions.head()