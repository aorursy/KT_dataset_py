import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split

import keras

from keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing import image



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import pickle



from tensorflow.keras.optimizers import RMSprop

from keras import regularizers

from tensorflow.keras import Model

import tensorflow as tf 



from tensorflow.keras import optimizers



import itertools



import tensorflow as tf 
base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}



sex_dict_enc = {

    'female' : 0,

    'male' : 1

}

loca_dict_enc = {

'scalp' : 0 , 'ear' : 1, 'face' : 2, 'back' : 3, 'trunk' : 4, 'chest' : 5,

       'upper extremity' : 6, 'abdomen' : 7, 'unknown' : 8, 'lower extremity' : 9,

       'genital' : 10, 'neck' : 11, 'hand' : 12, 'foot' : 13, 'acral' :  14}
metadata = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

metadata['path'] = metadata['image_id'].map(imageid_path_dict.get)

metadata['sex'] = metadata['sex'].map(sex_dict_enc.get)

metadata['localization'] = metadata['localization'].map(loca_dict_enc.get)



metadata = metadata[['path', 'dx', 'sex', 'age', 'localization']]



metadata.head()
metadata.isnull().sum()
metadata['age'].fillna((metadata['age'].mean()), inplace=True)

metadata['sex'].fillna(metadata['sex'].value_counts().index[0], inplace=True)

metadata.isnull().sum()
dxList = metadata.dx.unique()

dxDict = {}



for i in dxList:

    dxDict[i]= pd.DataFrame(metadata[metadata.dx == i])

    

dxDataTrain = pd.DataFrame(columns=metadata.keys())

dxDataTest = pd.DataFrame(columns=metadata.keys())

dxDataValid = pd.DataFrame(columns=metadata.keys())



for i in dxDict.keys():

    

    x_train, x_test = train_test_split(dxDict[i], test_size=0.17,random_state=1)

    

    print(i, len(x_train))

    x_train, x_valid = train_test_split(x_train, test_size = 0.2, random_state = 1)

   

    print(len(x_train), len(x_valid), len(x_test))

   

    dxDataTrain = pd.concat([dxDataTrain, x_train],axis=0).sample(frac=1).reset_index(drop=True)

    dxDataValid = pd.concat([dxDataValid, x_valid],axis=0).sample(frac=1).reset_index(drop=True)

    dxDataTest = pd.concat([dxDataTest, x_test],axis=0).sample(frac=1).reset_index(drop=True)

    

print(len(dxDataTrain), len(dxDataValid), len(dxDataTest))

dataGen = ImageDataGenerator(rescale=1./255, 

                             samplewise_center=True, 

                             samplewise_std_normalization=True, 

                             zoom_range = 0.1, 

                             width_shift_range=0.1, 

                             height_shift_range=0.1, 

                             horizontal_flip=True, 

                             vertical_flip=True) 

batch_size=64

    

trainDataGen = dataGen.flow_from_dataframe(dxDataTrain, x_col='path', y_col='dx', batch_size=batch_size ,target_size=(200, 200))

validDataGen = dataGen.flow_from_dataframe(dxDataValid, x_col='path', y_col='dx', batch_size=batch_size ,target_size=(200, 200))

testDataGen = dataGen.flow_from_dataframe(dxDataTest, x_col='path', y_col='dx', batch_size=batch_size ,target_size=(200, 200))

input_shape = (200, 200, 3)

num_classes = 7

preInceptionV3 = tf.keras.applications.InceptionV3(

    input_shape=input_shape, include_top=False, weights='imagenet')



# preMobileNetV2.summary()
def createModel(premodel):

    

    for layer in premodel.layers:

        layer.trainable = True

    last_layer = premodel.get_layer('mixed0')

    

    last_output = premodel.output 

    

    x = layers.Flatten()(last_output)

    x = layers.Dense(16, kernel_regularizer=regularizers.l1(0.0001), activation='relu')(x)

    x = layers.Dropout(0.2)(x) 

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense  (7, activation='softmax')(x)           



    model = Model(premodel.input, x) 



    opt = optimizers.Adam(lr=0.01)

    model.compile(loss = "categorical_crossentropy", optimizer = opt, 

                  metrics = ['accuracy'])

    

    return model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.callbacks import ReduceLROnPlateau



# Create a callback that saves the model's weights  ../input/

cp_path = "bestcp.ckpt"

cp_dir = os.path.dirname(cp_path)

cp_cb = ModelCheckpoint(filepath=cp_path,

                        save_weights_only=False, 

                        save_best_only=True, 

                        verbose=1)



# Adjcp_cbust learning rate based on number of GPUs (naive approach).

rlr_cb = ReduceLROnPlateau(monitor="val_accuracy",

                           factor=0.5,

                           patience=3,

                           verbose=1,

                           min_lr=0.001)



es_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)



# cp_cb, rlr_cb, es_cb
model = createModel(preInceptionV3)
epochs = 30

class_weights={

    0: 1.0, # akiec

    1: 1.0, # bcc

    2: 1.0, # bkl

    3: 1.0, # df

    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.

    5: 0.5, # nv

    6: 1.0, }# vasc 

    

history = model.fit_generator(

    trainDataGen,

    class_weight=class_weights,

    steps_per_epoch=len(trainDataGen),

    epochs=epochs,

    validation_data=validDataGen,

    validation_steps=len(validDataGen), 

    callbacks=[cp_cb,  es_cb, rlr_cb] )
# Get the labels that are associated with each index

print(trainDataGen.class_indices)

# get the metric names so we can use evaulate_generator

model.metrics_names
loss, accuracy = model.evaluate_generator(testDataGen, steps=len(testDataGen), verbose=1)

loss_v, accuracy_v = model.evaluate_generator(validDataGen, steps=len(validDataGen), verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

model.save("model.h5")



print("Saved model to disk")
# Here the best epoch will be used.



# model.load_weights(cp_path)

bestModel = tf.keras.models.load_model(cp_path)



loss, accuracy = bestModel.evaluate_generator(testDataGen, steps=len(testDataGen), verbose=1)

loss_v, accuracy_v = bestModel.evaluate_generator(validDataGen, steps=len(validDataGen), verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

# model.save("model.h5")



print("Saved model to disk")
# Get the labels of the test images.

test_labels = testDataGen.classes
predictions = model.predict_generator(testDataGen, steps=len(testDataGen), verbose=1)

bestPredictions = bestModel.predict_generator(testDataGen, steps=len(testDataGen), verbose=1)
# Source: Scikit Learn website

# http://scikit-learn.org/stable/auto_examples/

# model_selection/plot_confusion_matrix.html

#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py





def plotConfusionMatrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
from sklearn.metrics import confusion_matrix 

# argmax returns the index of the max value in a row

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

bestCm = confusion_matrix(test_labels, bestPredictions.argmax(axis=1))
# Define the labels of the class indices. These need to match the 

# order shown above.

plotConfusionMatrix(cm, dxDict.keys(), title='Confusion Matrix')
plotConfusionMatrix(bestCm, dxDict.keys(), title='bestConfusion Matrix')
# Get the index of the class with the highest probability score

y_pred = np.argmax(predictions, axis=1)



bestY_pred = np.argmax(bestPredictions, axis=1)



# Get the labels of the test images.

y_true = testDataGen.classes
from sklearn.metrics import classification_report



# Generate a classification report

report = classification_report(y_true, y_pred, target_names=dxDict.keys())

bestReport = classification_report(y_true, bestY_pred, target_names=dxDict.keys())

print(report, bestReport)
#1. Function to plot model's validation loss and validation accuracy

def plotModelHistory(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])

    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()

    

plotModelHistory(history)
t_img = image.load_img(dxDataTrain['path'].iloc[0], target_size=(200, 200))

t_img = image.img_to_array(t_img)

t_img = np.expand_dims(t_img, axis=0)                          
classes = model.predict(t_img)

classes