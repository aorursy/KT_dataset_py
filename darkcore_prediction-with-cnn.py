import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from warnings import filterwarnings

filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_data.head()
test_data.head()
train_data.info()
pd.get_dummies(train_data["label"]).values
train_data_labels = pd.get_dummies(train_data["label"]).values

train_data_pixels = (train_data.iloc[:,1:].values)/255.0

test_data_pixels = (test_data.values)/255.0
train_data_labels
train_data_pixels = train_data_pixels.reshape(-1,28,28,1)

test_data_pixels = test_data_pixels.reshape(-1,28,28,1)

#train_data_pixels[:1]
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score



x_train,x_test,y_train,y_test = train_test_split(train_data_pixels,train_data_labels,test_size=0.2,random_state=42)
x_train.shape
y_train.shape
fig, ax = plt.subplots(5,5,sharex=True,sharey=True)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=1)

for f in range(5):

    for g in range(5):

        ranran = np.random.randint(0,len(x_train))

        ax[f,g].imshow(x_train[ranran].reshape(28,28),cmap="gray")

        ax[f,g].set_title("Label:{}".format(y_train[ranran].argmax(-1)))

        ax[f,g].axis("off")

        ax[f,g].axis("tight")

        ax[f,g].axis("image")
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dropout, SpatialDropout2D, BatchNormalization, Input, Activation, Dense, Flatten

from keras.optimizers import Adam, RMSprop

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model



def build_model():

        model = Sequential()

        

        model.add(Input(shape=(28,28,1,)))



        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        model.add(Conv2D(filters=64,kernel_size=(6,6),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        

        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        model.add(Conv2D(filters=128,kernel_size=(6,6),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        

        model.add(Conv2D(filters=128,kernel_size=(3,3),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        model.add(MaxPool2D(pool_size=(2,2)))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        model.add(Conv2D(filters=256,kernel_size=(6,6),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        

        model.add(Conv2D(filters=256,kernel_size=(3,3),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(filters=32,kernel_size=(2,2),padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(SpatialDropout2D(0.25))

        

        

        model.add(Flatten())

        

        model.add(Dense(256))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.25))

        

        model.add(Dense(10))

        model.add(Activation("softmax"))

        

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        model.compile(optimizer = optimizer ,metrics=["accuracy"], loss = "categorical_crossentropy")

        

        return model
model = build_model()

plot_model(model,show_shapes=True,show_layer_names=True)
train_image_generator = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    zca_epsilon=1e-06,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    brightness_range=None,

    shear_range=0.1,

    zoom_range=0.1,

    channel_shift_range=0.0,

    fill_mode="nearest",

    cval=0.0,

    horizontal_flip=False,

    vertical_flip=False,

    rescale=None,

    preprocessing_function=None,

    data_format=None,

    validation_split=0.0,

    dtype=None,

)

train_image_generator.fit(x_train)
callback_lr = ReduceLROnPlateau(monitor='val_acc',patience=2,factor=0.5,min_lr=0.00001,verbose=1)

epochs = 50

batch_size = 256

history = model.fit_generator(train_image_generator.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[callback_lr])
model.fit(x_test,y_test,epochs=10,batch_size=512,validation_data=(x_train,y_train),callbacks=[callback_lr])
ypred = model.predict(test_data_pixels)

ypred = ypred.argmax(-1)

subimageid = [x for x in range(1,len(test_data_pixels)+1)]

submission = pd.DataFrame({"ImageId":subimageid,"Label":ypred})

submission.to_csv("submission.csv",index=False)