#Importing Librarys

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications.xception import Xception

from sklearn.metrics import classification_report, roc_curve, confusion_matrix

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

import cv2
trainDir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/'

testDir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'

valDir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/' 



width= 224

height = 224

batch_size = 16





trainGen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=7,

    zoom_range=0.3,

    width_shift_range=0.2,

    height_shift_range=0.2

)



valGen = ImageDataGenerator(

    rescale=1./255

)



trainGenerator = trainGen.flow_from_directory(

    trainDir,

    target_size = (width, height),

    batch_size = batch_size,

    class_mode = 'binary',

    color_mode = 'rgb'

)



valGenerator = valGen.flow_from_directory(

    valDir,

    target_size = (width, height),

    batch_size = batch_size,

    class_mode = 'binary',

    color_mode = 'rgb',

    shuffle = False

)





from tensorflow.keras.applications.xception import Xception



xceptionConv = Xception(include_top=False, weights="/kaggle/input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", input_shape=(224,224,3))
xceptionConv = Xception(include_top=False, weights="/kaggle/input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", input_shape=(224,224,3))



x = Flatten()(xceptionConv.output)

x = Dropout(0.5)(x)

x = Dense(512, activation = 'relu')(x)

x = Dropout(0.5)(x)

x = Dense(512, activation = 'relu')(x)

output = Dense(1, activation = 'sigmoid')(x)



xceptionModel = Model(inputs = xceptionConv.input, outputs = output)





xceptionModel.compile(

    optimizer=Adam(),

    loss='binary_crossentropy',

    metrics=['accuracy']

)



xceptionModel.summary()
stepsValidation = valGenerator.samples // batch_size

stepsTraining = trainGenerator.samples // batch_size



xceptionEarlyStopping = EarlyStopping(

    monitor='val_accuracy',

    mode='auto', 

    baseline=None, 

    restore_best_weights=True, 

    patience = 8,

    verbose = 1

)



checkpoint = ModelCheckpoint('top_layers.xception.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')





xceptionHistory = xceptionModel.fit_generator(

    generator = trainGenerator,

    steps_per_epoch = stepsTraining,

    epochs=40,

    validation_data = valGenerator,

    validation_steps = stepsValidation,

    callbacks=[checkpoint, xceptionEarlyStopping]

)
xceptionModel.load_weights('top_layers.xception.hdf5')
def predictClasses(predictions):

    valResult = predictions.copy()

    valResult[valResult <= 0.5] = 0

    valResult[valResult > 0.5] = 1

    return valResult



def plotHistory(history):

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy history')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss history')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



def plotConfusionMatrix(predictions, labels):

    valResult = predictClasses(predictions)

    confMatrixdf = pd.DataFrame(

        confusion_matrix(labels, valResult),

        index=["Exp. Normal", "Exp. Pneumonia"],

        columns=["Pred. Normal", "Pred. Pneumonia"]

    )



    plt.title("Confusion Matrix")

    sns.heatmap(confMatrixdf, annot=True, annot_kws={"size" : "20"})
plotHistory(xceptionHistory)
xceptionPrediction = xceptionModel.predict_generator(valGenerator, stepsValidation)



plotConfusionMatrix(

    xceptionPrediction, 

    valGenerator.classes

)
print(

    "\nReport: \n", 

    classification_report(

        valGenerator.classes, 

        predictClasses(xceptionPrediction), 

        target_names = ["Normal", "Pneumonia"]

    )

)