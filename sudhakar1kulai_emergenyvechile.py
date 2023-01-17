import numpy as np 

import pandas as pd 

import os

from keras import layers

from keras import models

from keras.utils import to_categorical

import matplotlib.pyplot as plt

from os import listdir, makedirs

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16, ResNet50, VGG19, InceptionV3

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout

from keras import optimizers, regularizers

from keras.optimizers import SGD

from glob import glob

import cv2

import warnings 

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline



import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn import metrics



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras.losses import categorical_crossentropy

from keras.models import Model

from keras.models import load_model

from keras.preprocessing import image

from mpl_toolkits.axes_grid1 import ImageGrid



import tensorflow as tf

import time

import os

#from tqdm import tqdm # for progress indication



print(os.listdir("../input"))

data_dir = '../input/'

data_dir1= '../input/state-farm-distracted-driver-detection/'
right_root_dir = '../input/right-cure-final/imgs_right_cure_final_noise'



left_root_dir= '../input/state-farm-distracted-driver-detection/'
#right_root_dir = '../input/right_created/'
class_list =  ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6','c7', 'c8', 'c9']

left_class_desc = ['safe driving', 'texting-right', 'talking on the phone-right', 'texting-left', 'talking on the phone-left', 

              'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']

right_class_desc = ['safe driving', 'texting-left', 'talking on the phone-left', 'texting-right', 'talking on the phone-right', 

              'operating the radio', 'drinking', 'reaching behind', 'hair and makeup', 'talking to passenger']

df_desc = pd.DataFrame({'class': class_list, 'left_desc': left_class_desc,  'right_desc': right_class_desc})

df_desc
left_train_dir = os.path.join(left_root_dir, 'train')

left_test_dir = os.path.join(left_root_dir, 'test')



right_train_dir = os.path.join(right_root_dir, 'train')

right_test_dir = os.path.join(right_root_dir, 'test')
class_dirs = os.listdir(right_train_dir)

for classname in class_dirs:

    if classname != '.DS_Store':

        print('{}: {} images'.format(classname, len(os.listdir(os.path.join(right_train_dir, classname)))))
train = []

for class_id, classname in enumerate(class_list):

    for file in os.listdir(os.path.join(right_train_dir, classname)):

        train.append(['train/{}/{}'.format(classname, file), class_id, classname])

        

train = pd.DataFrame(train, columns=['file', 'class_id', 'classname'])

train.head(2)
num_classes = len(np.unique(train["class_id"]))

print("num of classes: ", num_classes)
from sklearn.model_selection  import train_test_split

target = train["class_id"]



# split dataset into training and validation data with 70:30 split

test_size = 0.30 # taking 70:30 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=test_size, random_state=seed)



# pls note X_train & X_test contain the records of image file.. later we will read these images

# and converted into image data(i.e pixel) for further processing
print("Training shape: {}".format(X_train.shape))

print("Validation shape: {}".format(X_val.shape))
import cv2

from keras.preprocessing import image

def read_img_cv2_gray(filepath, size):

    img = cv2.imread(os.path.join(right_root_dir, filepath)) #, cv2.IMREAD_GRAYSCALE

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #IMREAD_GRAYSCALE

    resizeImg = cv2.resize(gray, size, interpolation = cv2.INTER_AREA) # resize image  

    img_data = image.img_to_array(resizeImg)

    img_data = np.expand_dims(img_data.copy(), axis=0)

    return img_data



def read_img_cv2(filepath, size):

    img = cv2.imread(os.path.join(right_root_dir, filepath)) #, cv2.IMREAD_GRAYSCALE

    resizeImg = cv2.resize(img, size, interpolation = cv2.INTER_AREA) # resize image  

    img_data = image.img_to_array(resizeImg)

    img_data = np.expand_dims(img_data.copy(), axis=0)

    return img_data
INPUT_SIZE = 128  # to experiment with higher pixel size



# reading image file for traing dataset

X_train_features = np.zeros((len(X_train), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, file in enumerate(X_train['file']):

    img_data = read_img_cv2(file, (INPUT_SIZE, INPUT_SIZE))

    X_train_features[i] = img_data

print('Training Images shape: {} size: {:,}'.format(X_train_features.shape, X_train_features.size))



# reading image file for validation dataset

X_val_features = np.zeros((len(X_val), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, file in enumerate(X_val['file']):

    img_data = read_img_cv2(file, (INPUT_SIZE, INPUT_SIZE))

    X_val_features[i] = img_data

print('Validation Images shape: {} size: {:,}'.format(X_val_features.shape, X_val_features.size))
X_train_features = X_train_features.astype('float32')/255

X_val_features = X_val_features.astype('float32')/255


y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)

y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)
print("Training data shape:")

print("Features: ",X_train_features.shape)

print("Target: ",y_train_cat.shape)
print("\nValidation data shape:")

print("Features: ",X_val_features.shape)

print("Target: ",y_val_cat.shape)


# utility fxn to plot model history and accuracy for each epoch

def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

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

    

# utiliy fxn to get y_predict in 1D

# y_predict is array of 12 classes for each cases.. let form the new data which give label value in 1D.. 

# this is required for classification matrix.. cm expect 1D array

def get1D_y_predict(y_pred):

    result = []

    for i in range(len(y_pred)):

        result.append(np.where(y_pred[i] == np.max(y_pred[i]))[0][0])

    return result    



def plot_cnf_matrix(cnf_matrix, name):

    fig, ax = plt.subplots(1, figsize=(12,5))

    ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)

    ax.set_xticklabels(class_list)

    ax.set_yticklabels(class_list)

    plt.title('Confusion Matrix')

    plt.ylabel('True class')

    plt.xlabel('Predicted class')

    fig.savefig('{}_cnf.png'.format(name), dpi=300)

    plt.show();

    

# use tensorboard callback which will passed in model.fit function.

# utility fxn ffor Initializing Early stopping and Model chekpoint callbacks**

def EarlyStopingModelCheckPoint():

    #tensorboard = TensorBoard(log_dir=".logs/{}".format(time.time()))



    #Adding Early stopping callback to the fit function is going to stop the training,

    #if the val_loss is not going to change even '0.001' for more than 5 continous epochs



    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)



    #Adding Model Checkpoint callback to the fit function is going to save the weights whenever val_loss achieves 

    # a new low value. Hence saving the best weights occurred during training



    model_checkpoint =  ModelCheckpoint('bestmodel.h5',

                                                               monitor='val_loss',

                                                               verbose=1,

                                                               save_best_only=True,

                                                               save_weights_only=False,

                                                               mode='auto',

                                                               period=1)

    return early_stopping, model_checkpoint
def create_model_resnet():

    resnet = ResNet50(include_top=False, input_shape=(224, 224, 3))

    

    model = Sequential()

    model.add(resnet)

    model.add(Flatten())

    model.add(Dense(1024, activation = "relu"))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(Dense(10, activation='softmax'))   

    model.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    return resnet, model
def create_my_resnet(resnet):    

    model = Sequential()

    model.add(resnet)

    model.add(Flatten())

    model.add(Dense(1024, activation = "relu"))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dropout(0.5))

    model.add(Dense(10, activation='softmax'))   

    model.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    return  model
resnet = ResNet50(include_top=False, input_shape=(128, 128, 3))

resnet.summary()
# Freeze the layers except the last 4 layers

#for layer in resnet.layers[:-4]:

    #layer.trainable = False

 

# Check the trainable status of the individual layers

#for layer in resnet.layers:

    #print(layer, layer.trainable)
model_resnet = create_my_resnet(resnet)
early_stopping,model_checkpoint = EarlyStopingModelCheckPoint()



# Train the model

start = time.time()



history1 = model_resnet.fit(X_train_features, y_train_cat, # feature and target vector

          validation_data=(X_val_features, y_val_cat), # data for evaluation

          epochs=50, #200

          batch_size=64, # Number of observations per batch

          verbose=1,     # Print description after each epoch

          callbacks=[early_stopping,model_checkpoint])



end = time.time()

execution_dur1 = end - start;
# plot model history

plot_model_history(history1)



# compute accuracy for validation dataset

val_loss, val_acc = model_resnet.evaluate(X_val_features, y_val_cat)

print('\nValidation accuracy: %0.2f' %(val_acc*100))
# predict the model

y_predict = model_resnet.predict(X_val_features)

y_predict1D = get1D_y_predict(y_predict)

acc = metrics.accuracy_score(y_val,y_predict1D)

print('Validation accuracy: %0.2f' %(acc*100))



# classification report with model acciracy and F1 score

cr = metrics.classification_report(y_val,y_predict1D)

print("Classification Report: \n\n", cr)