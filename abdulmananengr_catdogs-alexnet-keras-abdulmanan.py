# extracting files
import os, shutil, zipfile

folders = ['train', 'test1']

for fn in folders:
    with zipfile.ZipFile('../input/dogs-vs-cats/' + fn + ".zip", "r") as z:
        z.extractall(".")


#move cat images to /train/cat/ and dog images to /train/dog/
train_folder = "train/"
test_folder = "test1/"

!mkdir './train/cat'
!mkdir './train/dog'
!bash -c 'mv ./train/cat*.jpg ./train/cat/'
!bash -c 'mv ./train/dog*.jpg ./train/dog/'

#create data genetator
from keras.preprocessing.image import ImageDataGenerator
train_dog_dir = train_folder + "dog/"
train_cat_dir = train_folder + "cat/"

trainGen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

testGen = ImageDataGenerator(rescale=1./255)

#create iterators
train_it = trainGen.flow_from_directory(train_folder,
     batch_size=32, target_size=(100, 100),subset='training')

val_it = trainGen.flow_from_directory(train_folder, batch_size=32, 
    target_size=(100, 100),subset='validation')

test_it = testGen.flow_from_directory('test1/', batch_size=32, target_size=(100, 100))

#build model

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.optimizers import Adam

def build_model(width, height, depth, classes, reg=0.0002):

        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
       
        # Block #1: first CONV => RELU => POOL layer set
        model.add(Conv2D(96, (11, 11), strides=(4, 4),
            input_shape=inputShape, padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(256, (5, 5), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
    

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
    
        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block #5: second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))
        
        opt = Adam(lr=1e-3)
        model.compile(loss="binary_crossentropy", optimizer=opt,
            metrics=["accuracy"])


        # return network architecture
        return model
    
    

model = build_model(width=100, height=100, depth=3,
    classes=2, reg=0.0002)
model.summary()
H = model.fit_generator(train_it,
                         steps_per_epoch=train_it.samples // 32, 
                         epochs=10,
                         validation_data=val_it,
                         validation_steps=val_it.samples // 32)
# save the network to disk
print("serializing network...")
model.save("cat_dog_alexnet.hdf5")
import matplotlib.pyplot as plt
import numpy as np

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()