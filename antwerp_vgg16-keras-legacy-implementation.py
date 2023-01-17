import numpy as np

from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import VGG16

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, Activation, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator
classes = ["men", "women"]

nb_classes = len(classes)

batch_size = 32

nb_epoch = 30



img_rows, img_cols = 224, 224
def build_model():

    input_tensor = Input(shape=(img_rows, img_cols, 3))

    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)

    

    _model = Sequential()

    _model.add(Flatten(input_shape=vgg16.output_shape[1:]))

    _model.add(Dense(256, activation="relu"))

    _model.add(Dropout(0.5))

    _model.add(Dense(nb_classes, activation="softmax"))

    model = Model(inputs=vgg16.input, outputs=_model(vgg16.output))

    

    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-4, momentum=0.9), metrics=["accuracy"])

    

    return model
datadir = "../input/men-women-classification/data/"



train_datagen = ImageDataGenerator(

    rescale=1.0 / 255, # pixel scaling

    shear_range=0.2, # stretch the image diagonally (pi/x)

    zoom_range=0.2, # zoom randomly

    horizontal_flip=True, # randomly rotate horizontally

    validation_split=0.2 # set validation split

)



train_generator = train_datagen.flow_from_directory(

    datadir,

    target_size=(img_rows, img_cols),

    color_mode="rgb",

    classes=classes,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    subset="training" # set as training data

)



validation_generator = train_datagen.flow_from_directory(

    datadir,

    target_size=(img_rows, img_cols),

    color_mode="rgb",

    classes=classes,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    subset="validation" # set as validation data

)
model = build_model()

history = model.fit_generator(

    train_generator,

    epochs=nb_epoch,

    validation_data=validation_generator

)
def result(history):

    import matplotlib.pyplot as plt

    

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(acc))



    # accracy plot

    plt.plot(epochs, acc, 'bo' ,label = 'training acc')

    plt.plot(epochs, val_acc, 'b' , label= 'validation acc')

    plt.title('Training and Validation acc')

    plt.legend()



    plt.figure()



    # loss plot

    plt.plot(epochs, loss, 'bo' ,label = 'training loss')

    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')

    plt.title('Training and Validation loss')

    plt.legend()



    plt.show()
result(history)
# hdf5_file = "flower-model.hdf5"

# model.save_weights(hdf5_file)