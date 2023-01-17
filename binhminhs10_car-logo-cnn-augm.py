

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np



# Importing scikit-learn tools

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Importing standard ML set - numpy, pandas, matplotlib

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec



# Importing keras and its deep learning tools - neural network model, layers, contraints, optimizers, callbacks and utilities

from keras.models import Sequential, Model

from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.constraints import maxnorm

from keras.optimizers import Adam, RMSprop, SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.utils import np_utils

from keras.regularizers import l2

from keras.initializers import RandomNormal, VarianceScaling



image_path = '../input/images/images/'

imgs = os.listdir(image_path)

img_x = img_y = 50 # image size is constant

n_samples = np.size(imgs)

n_samples # 20778 originally

from PIL import Image

# loading all images

images = np.array([ np.array( Image.open(image_path+img).convert("RGB") ).flatten() for img in imgs], order='F', dtype='uint8')

# Mỗi ảnh có kích thước 50x50 = 2500 pixel và 3 kênh màu = 7500 pixel

print('total images: ', np.shape(images) )

# Producing label and assigning them accordingly

import re

cars = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge',

        'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada',

        'Lancia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi',

        'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Saab', 'Seat',

        'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']

# re.match()[0] lấy về tên car , car.index trả về index ứng với tên car vd: Daewoo index là 6 (cars[6] = 'Daewoo')

labels = np.array([ cars.index(re.match(r"(^\D+)", imgs[i])[0]) for i in range(n_samples)])

print('total label images: ', labels.shape )

labels_pd = pd.DataFrame(labels)

labels_pd[0].value_counts()
# preparation data

dataset, labelset = shuffle(images, labels, random_state=42)

train_data = [dataset, labelset]

# an example image

r=1260

plt.imshow(images[r].reshape(img_x, img_y, 3))

plt.title(cars[labels[r]])

plt.show()

# Training and preparing dataset

X_train, X_test, y_train, y_test = train_test_split( train_data[0], train_data[1], test_size=0.2)

# Maintain a copy of testset

X_test_img = X_test.copy()
# bring images back size (20778, 50, 50,3)

def ImageConvert(n, i):

    im_ex = i.reshape(n, img_x, img_y, 3)

    im_ex = im_ex.astype('float32') / 255

    # zero center data

    im_ex = np.subtract(im_ex, 0.5)

    # ...and to scale it to (-1, 1)

    im_ex = np.multiply(im_ex, 2.0)

    return im_ex

X_train = ImageConvert(X_train.shape[0], X_train)

X_test = ImageConvert(X_test.shape[0], X_test)



# Labels have to be transformed to categorical

Y_train = np_utils.to_categorical(y_train, num_classes=len(cars))

Y_test = np_utils.to_categorical(y_test, num_classes=len(cars))

# Four Conv/MaxPool blocks, a flattening layer and two dense layers at the end

def contruction(n_channels):

    model = Sequential()

    model.add(Conv2D(32, (3,3),

                     input_shape=(img_x,img_y,n_channels),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(64, (3,3),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(128, (3,3),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Conv2D(256, (3,3),

                     padding='valid',

                     bias_initializer='glorot_uniform',

                     kernel_regularizer=l2(0.00004),

                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    

    model.add(Flatten())

    

    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))

    model.add(Dropout(0.5))

    

    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))

    model.add(Dropout(0.5))

    

    # final activation is softmax, tuned to the number of classes/labels possible

    model.add(Dense(len(cars), activation='softmax'))

    

    # optimizer will be a stochastic gradient descent, learning rate set at 0.005

    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.95, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    return model

model = contruction(3)

# Let's look at the summary

model.summary()
# Some callbacks have to be provided to choose the best trained model

# patience set at 4 as 3 was too greedy - I observed better results after the third-worse epoch

early_stopping = EarlyStopping(patience=6, monitor='val_loss')

CNN_file = 'car_CNN_9AUGM_CMCMCMCMF.h5py' # the 13th try, with augmented data

take_best_model = ModelCheckpoint(CNN_file, save_best_only=True)
from keras.preprocessing.image import ImageDataGenerator



image_gen = ImageDataGenerator(

    #featurewise_center=True,

    #featurewise_std_normalization=True,

    rotation_range=45,

    width_shift_range=.15,

    height_shift_range=.15,

    horizontal_flip=True,

    vertical_flip=True)



#training the image preprocessing

image_gen.fit(X_train, augment=True)

NUM_EPOCHS = 100

BATCH_SIZE = 128





# monitor the validation accuracy, reduce the learning rate by factor when there is no improvement after the number of patience 

reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', 

                              factor=0.3, 

                              patience=3, 

                              min_lr=0.0001)



callbacks_list = [reduce_lr, early_stopping, take_best_model]



history = model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

                              steps_per_epoch=X_train.shape[0]//BATCH_SIZE,

                              epochs=NUM_EPOCHS,

                              verbose=1,

                              validation_data=(X_test, Y_test),

                              callbacks=callbacks_list)

# model.save_weights("car_CNN_AUGM_CMCMCMCMF.h5")

# Plot the training and validation loss + accuracy

def plot_training(history):

    acc = history.history['categorical_accuracy']

    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(acc))



    plt.plot(epochs, acc, 'r.')

    plt.plot(epochs, val_acc, 'r')

    plt.title('Training and validation accuracy')



    # plt.figure()

    # plt.plot(epochs, loss, 'r.')

    # plt.plot(epochs, val_loss, 'r-')

    # plt.title('Training and validation loss')

    plt.show()

    

plot_training(history)



pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]



#print("Saved model to disk")

scores = model.evaluate(X_test, Y_test) # let's look at the accuracy on the test set

print("Accuracy test: %.2f%%" % (scores[1]*100))
import os

print(os.listdir())
from sklearn.metrics import precision_recall_fscore_support as prfs



# Preparing for metrics check-up on the test set, may take a while...

Y_pred = model.predict_classes(X_test)
precision, recall, f1, support = prfs(y_test, Y_pred, average='weighted')

print("Precision: {:.2%}\nRecall: {:.2%}\nF1 score: {:.2%}\nAccuracy: {:.2%}".format(precision, recall, f1, scores[1]))
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import seaborn as sns #for better and easier plots



def report_and_confusion_matrix(label, prediction):

    print("Model Report")

    print(classification_report(label, prediction))

    score = accuracy_score(label, prediction)

    print("Accuracy : "+ str(score))

    

    ####################

    fig, ax = plt.subplots(figsize=(8,8)) #setting the figure size and ax

    mtx = confusion_matrix(label, prediction)

    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=True, ax=ax) #create a heatmap with the values of our confusion matrix

    plt.ylabel('true label')

    plt.xlabel('predicted label')



report_and_confusion_matrix(y_test, Y_pred)
# But let's check per class, too - assuming that larger datasets will be having higher metrics

precision_, recall_, f1_, support_ = prfs(y_test, Y_pred, average=None)

# We see that smaller sets (Lexus, Jaguar, Hyundai) have generally worse precision and recall

plt.subplots(figsize=(18,30))

x = range(len(cars))

plt.subplot(311)

plt.title('Precision per class')

plt.ylim(0.5, 1.00)

plt.bar(x, precision_, color='Red')

plt.xticks(x, cars, rotation = 90)

plt.subplot(312)

plt.title('Recall per class')

plt.ylim(0.5, 1.00)

plt.bar(x, recall_, color='Green')

plt.xticks(x, cars, rotation = 90)

plt.subplot(313)

plt.title('F1 score per class')

plt.ylim(0.5, 1.00)

plt.bar(x, f1_, color='Blue')

plt.xticks(x, cars, rotation = 90)

plt.show()
# OK, let's try the CNN in action - first defining the ShowCase() method to show everything nicely



def ShowCase(cols, rows):

    fdict = {'fontsize': 24,

            'fontweight' : 'normal',

            'verticalalignment': 'baseline'}

    plt.figure(figsize=(cols * 5, rows * 4))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    c = 0

    for i in range(rows * cols):

        plt.subplot(rows, cols, i + 1)

        

        # r - randomly picked from the whole dataset

        r = np.random.randint(np.shape(images)[0])

        

        # j - predicted class for the image of index r (weird syntax, but works :)

        j = int(model.predict_classes(ImageConvert(1, images[r:r+1]), verbose=0))

        

        # increase success if predicted well

        if labels[r] == j:

            c += 1

        

        # image needs reshaping back to a 50px*50px*RGB

        plt.imshow(images[r].reshape(img_x, img_y, 3))

        

        # plt.title will show the true brand and the predicted brand

        plt.title('True brand: '+cars[labels[r]]+'\nPredicted: '+cars[j],

                  color= 'Green' if cars[labels[r]] == cars[j] else 'Red', fontdict=fdict) # Green for right, Red for wrong

        

        # no ticks

        plt.xticks(())

        plt.yticks(())

        

    # print out the success rate

    print('Success rate: {}/{} ({:.2%})'.format(c, rows*cols, c/(rows*cols)))

    

    plt.show()


# That is strictly for the showcasing, how the CNN works - ain't that bad, after all :)

ShowCase(10, 5)
print(os.listdir())
new_image_path = '../input/images/new_images/'

new_imgs = os.listdir(new_image_path)

new_n_samples = np.size(new_imgs)

new_n_samples # 8
cols = 8

rows = 1

plt.figure(figsize=(cols * 5, rows * 4))

plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

for i in range(new_n_samples):

    plt.subplot(rows, cols, i + 1)

    im = Image.open(new_image_path+new_imgs[i]).convert("RGB")

    new_im = np.array(im.resize((50,50))).flatten()

    m = int(model.predict_classes(ImageConvert(1, new_im), verbose=0))

    plt.imshow(new_im.reshape(img_x, img_y, 3))

    plt.title('Predicted brand: '+cars[m], size=24)

    plt.xticks(())

    plt.yticks(())

plt.show() # 5/8
from keras.applications.resnet50 import ResNet50, preprocess_input



HEIGHT = 50

WIDTH = 50

base_model = ResNet50(weights='imagenet', 

                      include_top=False, 

                    input_shape=(HEIGHT, WIDTH, 3))
from keras.preprocessing.image import ImageDataGenerator



image_gen = ImageDataGenerator(

    #featurewise_center=True,

    #preprocessing_function=preprocess_input,

    rotation_range=45,

    width_shift_range=.15,

    height_shift_range=.15,

    horizontal_flip=True,

    vertical_flip=True)



#training the image preprocessing

image_gen.fit(X_train)
def build_finetune_model(base_model, dropout, fc_layers, num_classes):

#     for layer in base_model.layers[:13]:

#         layer.trainable = False

        

    x = base_model.output

    x = Flatten()(x)

    for fc in fc_layers:

        # New FC layer, random init

        x = Dense(fc, activation='relu')(x) 

        x = Dropout(dropout)(x)



    # New softmax layer

    predictions = Dense(num_classes, activation='softmax')(x) 

    

    finetune_model = Model(inputs=base_model.input, outputs=predictions)



    return finetune_model





FC_LAYERS = [1024, 1024]

dropout = 0.5



finetune_model = build_finetune_model(base_model, 

                                      dropout=dropout, 

                                      fc_layers=FC_LAYERS, 

                                        num_classes=len(cars))
NUM_EPOCHS = 10

BATCH_SIZE = 128





adam = Adam(lr=0.00001)

sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.95, nesterov=True)

finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])





# Some callbacks have to be provided to choose the best trained model

# patience set at 4 as 3 was too greedy - I observed better results after the third-worse epoch

early_stopping = EarlyStopping(patience=8, monitor='val_loss')

ResNet_file = 'car_ResNet_AUGM.h5py' # the 13th try, with augmented data

take_best_model = ModelCheckpoint(ResNet_file, save_best_only=True)





filepath="ResNet50" + "_model_weights.h5"

checkpoint = ModelCheckpoint(filepath, monitor=["categorical_accuracy"], verbose=1, mode='max')



callbacks_list = [take_best_model,early_stopping]





callbacks_list = [early_stopping, take_best_model]



fitted_model = finetune_model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

                                          steps_per_epoch=X_train.shape[0]//BATCH_SIZE,

                                          epochs=NUM_EPOCHS,

                                          shuffle=True,

                                          verbose=1,

                                          validation_data=(X_test, Y_test),

                                          callbacks=callbacks_list)



pd.DataFrame(fitted_model.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D





# create the base pre-trained model

base_model = ResNet50(weights='imagenet', include_top=False)



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(len(cars), activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional ResNet layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



NUM_EPOCHS = 5

BATCH_SIZE = 128

# train the model on the new data for a few epochs

model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

                                  steps_per_epoch=len(X_train)//BATCH_SIZE,

                                  epochs=NUM_EPOCHS)

# let's visualize layer names and layer indices to see how many layers

# we should freeze:

for i, layer in enumerate(base_model.layers):

   print(i, layer.name)
# at this point, the top layers are well trained and we can start fine-tuning

# convolutional layers from ResNet. We will freeze the bottom N layers

# and train the remaining top layers.



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 249 layers and unfreeze the rest:

# for layer in model.layers[:249]:

#    layer.trainable = False

# for layer in model.layers[249:]:

#    layer.trainable = True

for layer in model.layers:

   layer.trainable = True



# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])



# Some callbacks have to be provided to choose the best trained model

# patience set at 4 as 3 was too greedy - I observed better results after the third-worse epoch

early_stopping = EarlyStopping(patience=8, monitor='val_loss')

InceptionV3_file = 'car_ResNet_AUGM.h5py' # the 13th try, with augmented data

take_best_model = ModelCheckpoint(InceptionV3_file, save_best_only=True)

callbacks_list = [take_best_model,early_stopping]



# we train our model again (this time fine-tuning the top 2 inception blocks

# alongside the top Dense layers

NUM_EPOCHS = 100

fitted_model2 = model.fit_generator(image_gen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

                                      steps_per_epoch=len(X_train)//BATCH_SIZE,

                                      epochs=NUM_EPOCHS,

                                      verbose=1,

                                      validation_data=(X_test, Y_test),

                                      callbacks=callbacks_list)
# Save the weights

model.save_weights('car_ResNet_AUGM_weights.h5')



# Save the model architecture

with open('model_car_ResNet_AUGM.json', 'w') as f:

    f.write(model.to_json())



#print("Saved model to disk")

scores = model.evaluate(X_test, Y_test) # let's look at the accuracy on the test set

print("Accuracy test: %.2f%%" % (scores[1]*100))



pd.DataFrame(fitted_model2.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 2) # set the vertical range to [0-1]

from sklearn.metrics import precision_recall_fscore_support as prfs

Y_pred = model.predict(X_test)

Y_pred = np.argmax(Y_pred,axis=1)



precision, recall, f1, support = prfs(y_test, Y_pred, average='weighted')

print("Precision: {:.2%}\nRecall: {:.2%}\nF1 score: {:.2%}\nAccuracy: {:.2%}".format(precision, recall, f1, scores[1]))
new_image_path = '../input/images/new_images/'

new_imgs = os.listdir(new_image_path)

new_n_samples = np.size(new_imgs)

new_n_samples # 8
cols = 8

rows = 1

plt.figure(figsize=(cols * 5, rows * 4))

plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

for i in range(new_n_samples):

    plt.subplot(rows, cols, i + 1)

    im = Image.open(new_image_path+new_imgs[i]).convert("RGB")

    new_im = np.array(im.resize((50,50))).flatten()

    m = int(np.argmax(model.predict(ImageConvert(1, new_im), verbose=0),axis=1))

    plt.imshow(new_im.reshape(img_x, img_y, 3))

    plt.title('Predicted brand: '+cars[m], size=24)

    plt.xticks(())

    plt.yticks(())

plt.show() 