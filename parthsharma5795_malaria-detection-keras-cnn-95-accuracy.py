import warnings

warnings.filterwarnings('ignore')

from __future__ import absolute_import, division, print_function

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from mlxtend.plotting import plot_confusion_matrix

from PIL import Image

import os

from keras.utils import to_categorical

from keras import backend as K

from keras import layers

from keras.preprocessing.image import save_img

from keras.utils.vis_utils import model_to_dot

from keras.models import Sequential,Input,Model

from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D

from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,SGD

from keras.utils.vis_utils import plot_model

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

print(os.listdir("../input/cell_images/cell_images"))
infected = os.listdir('../input/cell_images/cell_images/Parasitized/') 

uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')
data = []

labels = []



for i in infected:

    try:

    

        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((64 , 64))

        data.append(np.array(resize_img))

        label = to_categorical(1, num_classes=2)

        labels.append(label)

        

    except AttributeError:

        print('')

    

for u in uninfected:

    try:

        

        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+u)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((64 , 64))

        data.append(np.array(resize_img))

        label = to_categorical(0, num_classes=2)

        labels.append(label)

        

    except AttributeError:

        print('')
data = np.array(data)

labels = np.array(labels)



np.save('Data' , data)

np.save('Labels' , labels)
print('Cells : {} | labels : {}'.format(data.shape , labels.shape))
plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(data[0])

plt.title('Infected Cell')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(data[15000])

plt.title('Uninfected Cell')

plt.xticks([]) , plt.yticks([])



plt.show()
n = np.arange(data.shape[0])

np.random.shuffle(n)

data = data[n]

labels = labels[n]
data = data.astype(np.float32)

labels = labels.astype(np.int32)

data = data/255
from sklearn.model_selection import train_test_split



train_x , eval_x , train_y , eval_y = train_test_split(data , labels , 

                                            test_size = 0.2 ,

                                            random_state = 111)



# eval_x , test_x , eval_y , test_y = train_test_split(x , y , 

#                                                     test_size = 0.2 , 

#                                                     random_state = 111)
print('train data shape {} ,eval data shape {} '.format(train_x.shape, eval_x.shape))
train_aug = ImageDataGenerator(

    rescale=1./255,

    shear_range=0.2,  

    zoom_range=0.2,        

    horizontal_flip=True,

    vertical_flip=True)  



val_aug= ImageDataGenerator(

    rescale=1./255)



train_gen = train_aug.flow(

    train_x,

    train_y,

    batch_size=16)



val_gen = val_aug.flow(

    eval_x,

    eval_y,

    batch_size=16)

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
def ConvBlock(model, layers, filters,name):

    for i in range(layers):

        model.add(SeparableConv2D(filters, (3, 3), activation='relu',name=name))

        model.add(BatchNormalization())

        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    

def FCN():

    model = Sequential()

    model.add(Lambda(lambda x: x, input_shape=(64, 64, 3)))

    ConvBlock(model, 1, 64,'block_1')

    ConvBlock(model, 1, 128,'block_2')

    ConvBlock(model, 1, 256,'block_3')

    ConvBlock(model, 1, 512,'block_4')

    model.add(Flatten())

    model.add(Dense(1024,activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(2,activation='sigmoid'))

    return model



model = FCN()

model.summary()



# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
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

    patience=7,

    verbose=2,

    mode='min'

)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=40,

    verbose=1, 

    mode='auto',

    cooldown=1 

)



callbacks = [checkpoint,earlystop,reduce]
opt = SGD(lr=1e-4,momentum=0.99)

opt1 = Adam(lr=2e-4)



model.compile(

    loss='binary_crossentropy',

    optimizer=opt,

    metrics=['accuracy']

)

    

history = model.fit_generator(

    train_gen, 

    steps_per_epoch  = 5000, 

    validation_data  = val_gen,

    validation_steps = 2000,

    epochs = 10, 

    verbose = 1,

    callbacks=callbacks

)
show_final_history(history)

model.load_weights(best_model_weights)

model_score = model.evaluate_generator(val_gen,steps=50)

print("Model Test Loss:",model_score[0])

print("Model Test Accuracy:",model_score[1])

model.save('malaria.h5')
preds = model.predict(eval_x, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(eval_y, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)



print(np.unique(orig_test_labels))

print(np.unique(preds))


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(orig_test_labels , preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Infected'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Infected'], fontsize=16)

plt.show()
