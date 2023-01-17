%matplotlib inline

import pandas as pd

import os,shutil,math,scipy,cv2

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn 



from sklearn.utils import shuffle

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix,roc_curve,auc



from PIL import Image

from PIL import Image as pil_image

from time import time

from PIL import ImageDraw

from glob import glob

from tqdm import tqdm

from skimage.io import imread

from IPython.display import SVG



from scipy import misc,ndimage

from scipy.ndimage.interpolation import zoom

from scipy.ndimage import imread



from keras import backend as K

from keras import layers

from keras.preprocessing.image import save_img

from keras.utils.vis_utils import model_to_dot

from keras.applications.vgg19 import VGG19,preprocess_input

from keras.applications.xception import Xception

from keras.applications.nasnet import NASNetMobile

from keras.models import Sequential,Input,Model

from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D

from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,SGD

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
train_dir = '../input/seg_train/seg_train/'

test_dir = '../input/seg_test/seg_test/'
# Here's our 6 categories that we have to classify.

class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']

class_names_label = {'mountain': 0,

                    'street' : 1,

                    'glacier' : 2,

                    'buildings' : 3,

                    'sea' : 4,

                    'forest' : 5

                    }

nb_classes = 6
def load_data():    

    datasets = [train_dir, test_dir]

    size = (150,150)

    output = []

    for dataset in datasets:

        directory = "../input/" + dataset

        images = []

        labels = []

        for folder in os.listdir(directory):

            curr_label = class_names_label[folder]

            for file in os.listdir(directory + "/" + folder):

                img_path = directory + "/" + folder + "/" + file

                curr_img = cv2.imread(img_path)

                curr_img = cv2.resize(curr_img, size)

                images.append(curr_img)

                labels.append(curr_label)

        images, labels = shuffle(images, labels)     ### Shuffle the data !!!

        images = np.array(images, dtype = 'float32') ### Our images

        labels = np.array(labels, dtype = 'int32')   ### From 0 to num_classes-1!

        

        output.append((images, labels))



    return output
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images = train_images / 255.0 

test_images = test_images / 255.0
fig = plt.figure(figsize=(10,10))

fig.suptitle("Some examples of images of the dataset", fontsize=16)

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
augs = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,  

    zoom_range=0.2,        

    horizontal_flip=True,

    validation_split=0.3)  



train_gen = augs.flow_from_directory(

    train_dir,

    target_size = (150,150),

    batch_size=8,

    class_mode = 'categorical'

)



test_gen = augs.flow_from_directory(

    test_dir,

    target_size=(150,150),

    batch_size=8,

    class_mode='categorical'

)
def ConvBlock(model, layers, filters,name):

    for i in range(layers):

        model.add(SeparableConv2D(filters, (3, 3), activation='relu',name=name))

        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    

def FCN():

    model = Sequential()

    model.add(Lambda(lambda x: x, input_shape=(150, 150, 3)))

    ConvBlock(model, 1, 16,'block_1')

    ConvBlock(model, 1, 32,'block_2')

    ConvBlock(model, 1, 64,'block_3')

    ConvBlock(model, 1, 128,'block_4')

    model.add(Flatten())

    model.add(Dense(1024,activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(6,activation='sigmoid'))

    return model



model = FCN()

model.summary()



SVG(model_to_dot(model).create(prog='dot', format='svg'))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
#-------Callbacks-------------#

best_model_weights = './base.model'

checkpoint = ModelCheckpoint(

    best_model_weights,

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=10,

    verbose=1,

    mode='auto'

)

tensorboard = TensorBoard(

    log_dir = './logs',

    histogram_freq=0,

    batch_size=16,

    write_graph=True,

    write_grads=True,

    write_images=False,

)



csvlogger = CSVLogger(

    filename= "training_csv.log",

    separator = ",",

    append = False

)



#lrsched = LearningRateScheduler(step_decay,verbose=1)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=40,

    verbose=1, 

    mode='auto',

    cooldown=1 

)



callbacks = [checkpoint,tensorboard,csvlogger,reduce]
opt = SGD(lr=1e-4,momentum=0.99)

opt1 = Adam(lr=2e-4)



model.compile(

    loss='binary_crossentropy',

    optimizer=opt,

    metrics=['accuracy']

)



history = model.fit_generator(

    train_gen, 

    steps_per_epoch  = 500, 

    validation_data  = test_gen,

    validation_steps = 500,

    epochs = 20, 

    verbose = 1,

    callbacks=callbacks

)
def show_final_history(history):

    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('acc')

    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")

    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()
show_final_history(history)

model.load_weights(best_model_weights)

model_score = model.evaluate_generator(test_gen)

print("Model Test Loss:",model_score[0])

print("Model Test Accuracy:",model_score[1])
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):

    BOO = (test_labels == pred_labels)

    mislabeled_indices = np.where(BOO == 0)

    mislabeled_images = test_images[mislabeled_indices]

    mislabeled_labels = pred_labels[mislabeled_indices]

    fig = plt.figure(figsize=(10,10))

    fig.suptitle("Some examples of mislabeled images by the classifier:", fontsize=16)

    for i in range(25):

        plt.subplot(5,5,i+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(mislabeled_images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[mislabeled_labels[i]])

    plt.show()
predictions = model.predict(test_images)

pred_labels = np.argmax(predictions, axis = 1)

print_mislabeled_images(class_names, test_images, test_labels, pred_labels)