import os

print(os.listdir("../input/ckplus/ck/CK+48"))


imageSize=80

test_dir = '../input/ckplus/ck/CK+48/'



# ['DME', 'CNV', 'NORMAL', '.DS_Store', 'DRUSEN']

from tqdm import tqdm

def get_data(folder):

    """

    Load the data and labels from the given folder.

    """

    X = []

    y = []

    for folderName in os.listdir(folder):

        if not folderName.startswith('.'):

            if folderName in ['disgust']:

                label = 0

            elif folderName in ['anger']:

                label = 1

            elif folderName in ['sadness']:

                label = 2

            elif folderName in ['surprise']:

                label = 3

            elif folderName in ['contempt']:

                label = 4

            elif folderName in ['fear']:

                label = 5

            elif folderName in ['fear']:

                 label = 6

            else:

                label = 7



            for image_filename in tqdm(os.listdir(folder + folderName)):

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 1))

                    img_arr = np.asarray(img_file)

                    X.append(img_arr)

                    y.append(label)

    X = np.asarray(X) # Keras only accepts data as numpy arrays 

    y = np.asarray(y)

    return X,y

X_test, y_test = get_data(test_dir) # Un-comment to use full dataset: Step 1 of 2

#X_test, y_test= get_data(train_dir)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2) # comment this ligne  to use full dataset: Step 2 of 2

y_train.shape
import os

from glob import glob

import matplotlib.pyplot as plt

import random

import cv2

import pandas as pd

import numpy as np

import matplotlib.gridspec as gridspec

import seaborn as sns

import zlib

import itertools

import sklearn

import itertools

import scipy

import skimage

from skimage.transform import resize

import csv

from tqdm import tqdm

from sklearn import model_selection

from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold

from sklearn.utils import class_weight

from sklearn.metrics import confusion_matrix

import keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import models, layers, optimizers

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.utils import class_weight

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop

from keras.models import Sequential, model_from_json

from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50



data_path = '../input/ckplus/ck/CK+48'

labels = os.listdir('../input/ckplus/ck/CK+48')

train_datagen = ImageDataGenerator(samplewise_center=True, 

                              samplewise_std_normalization=True, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range= 0.05, 

                              width_shift_range=0.1, 

                              rotation_range=15, 

                              zoom_range=0.15,

                              validation_split=0.1)

IMG_SIZE = 224

batch_size = 36

train_data_dir = '../input/ckplus/ck/CK+48'

validation_data_dir = '../input/ckplus/ck/CK+48'

train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(IMG_SIZE , IMG_SIZE),

    batch_size=36,

    subset='training',

    class_mode='categorical')

valid_X, valid_Y = next(train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(IMG_SIZE , IMG_SIZE),

    batch_size=94,

    subset='validation',

    class_mode='categorical'))
t_x, t_y = next(train_generator)

fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone')

    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(labels, c_y) 

                             if n_score>0.5]))

    c_ax.axis('off')
input_shape=( 224, 224, 3)









model = Sequential()



model.add(Conv2D(32, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(4, 4)))



model.add(Conv2D(412, (5, 5), padding='same', activation = 'relu'))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(MaxPooling2D(pool_size=(3, 3)))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation = 'softmax'))





# Classification

# model.add(Flatten())

# model.add(Dense(64))

# model.add(Activation('relu'))

# model.add(Dropout(0.5))

# model.add(Dense(num_classes))

# model.add(Activation('softmax'))



#Compile Model

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
history = model.fit_generator(train_generator, 

                                  steps_per_epoch=887/36,

                                  validation_data = (valid_X,valid_Y), 

                                  epochs = 50

                                  )
score = model.evaluate(valid_X,valid_Y, verbose=0)

print('\n Model accuracy ON TEST SET :', score[1], '\n')

 
labelsFaces =['disgust', 'anger', 'sadness', 'surprise', 'contempt', 'fear', 'happy']





predictedExpression = model.predict(valid_X)



figure = plt.figure(figsize=(20, 8))



for i, index in enumerate(np.random.choice(valid_X.shape[0], size=25, replace=False)):

    ax = figure.add_subplot(5, 5, i + 1, xticks=[], yticks=[])

    # Display each image

    ax.imshow(np.squeeze(valid_X[index]))

    predict_index = np.argmax(predictedExpression[index])

    true_index = np.argmax(valid_Y[index])

    # Set the title for each image

    ax.set_title("{} ({})".format(labelsFaces[predict_index], 

                                  labelsFaces[true_index]),

                                  color=("green" if predict_index == true_index else "red"))

plt.show()

print(os.listdir("../input/jaffefacialexpression/jaffe/jaffe"))
data_path = '../input/jaffefacialexpression/jaffe/jaffe'

data_dir_list = os.listdir(data_path)



img_rows=256

img_cols=256

num_channel=1



num_epoch=10



img_data_list=[]





for dataset in data_dir_list:

    img_list=os.listdir(data_path+'/'+ dataset)

    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

    for img in img_list:

        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )

        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        input_img_resize=cv2.resize(input_img,(224,224))

        img_data_list.append(input_img_resize)

        

img_data = np.array(img_data_list)

img_data = img_data.astype('float32')

img_data = img_data/255

img_data.shape
num_classes = 7



num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,),dtype='int64')



labels[0:29]=0 #30

labels[30:58]=1 #29

labels[59:90]=2 #32

labels[91:121]=3 #31

labels[122:151]=4 #30

labels[152:182]=5 #31

labels[183:]=6 #30



names = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE']



def getLabel(id):

    return ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SURPRISE'][id]
Y = np_utils.to_categorical(labels, num_classes)

from sklearn.utils import shuffle



#Shuffle the dataset

x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

x_test=X_test

#X_train=X_train.reshape(X_train.shape[0],128,128,1)

#X_test=X_test.reshape(X_test.shape[0],128,128,1)
X_train.shape
from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization
input_shape=(224,224,3)



model = Sequential()



model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(258, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))



model.add(Dropout(0.6))

model.add(Dense(7, activation = 'softmax'))



 

#Compile Model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

model.summary()

model.get_config()

model.layers[0].get_config()

model.layers[0].input_shape

model.layers[0].output_shape

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable
from keras import callbacks

filename='model1_train_new.csv'

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"



csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]

callbacks_list = [csv_log]
hist = model.fit(X_train, y_train, batch_size=4, epochs=50, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
# visualizing losses and accuracy

%matplotlib inline



train_loss=hist.history['loss']

val_loss=hist.history['val_loss']

train_acc=hist.history['acc']

val_acc=hist.history['val_acc']



epochs = range(len(train_acc))



plt.plot(epochs,train_loss,'r', label='train_loss')

plt.plot(epochs,val_loss,'b', label='val_loss')

plt.title('train_loss vs val_loss')

plt.legend()

plt.figure()



plt.plot(epochs,train_acc,'r', label='train_acc')

plt.plot(epochs,val_acc,'b', label='val_acc')

plt.title('train_acc vs val_acc')

plt.legend()

plt.figure()
input_shape=(224,224,3)



model = Sequential()



model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.6))

model.add(Dense(7, activation = 'softmax'))





 

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
model.summary()

model.get_config()

model.layers[0].get_config()

model.layers[0].input_shape

model.layers[0].output_shape

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable
from keras import callbacks

filename='model_train_new.csv'

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"



csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,checkpoint]

callbacks_list = [csv_log]
hist = model.fit(X_train, y_train, batch_size=4, epochs=50, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
#Model Save

model.save_weights('model_weights.h5')

model.save('model_keras.h5')



# visualizing losses and accuracy

%matplotlib inline



train_loss=hist.history['loss']

val_loss=hist.history['val_loss']

train_acc=hist.history['acc']

val_acc=hist.history['val_acc']



epochs = range(len(train_acc))



plt.plot(epochs,train_loss,'r', label='train_loss')

plt.plot(epochs,val_loss,'b', label='val_loss')

plt.title('train_loss vs val_loss')

plt.legend()

plt.figure()



plt.plot(epochs,train_acc,'r', label='train_acc')

plt.plot(epochs,val_acc,'b', label='val_acc')

plt.title('train_acc vs val_acc')

plt.legend()

plt.figure()
# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)

print('Test Loss:', score[0])

print('Test accuracy:', score[1])



test_image = X_test[0:1]

print (test_image.shape)



print(model.predict(test_image))

print(model.predict_classes(test_image))

print(y_test[0:1])



res = model.predict_classes(X_test[9:18])

plt.figure(figsize=(10, 10))



for i in range(0, 9):

    plt.subplot(330 + 1 + i)

    plt.imshow(x_test[i],cmap=plt.get_cmap('gray'))

    plt.gca().get_xaxis().set_ticks([])

    plt.gca().get_yaxis().set_ticks([])

    plt.ylabel('prediction = %s' % getLabel(res[i]), fontsize=14)

# show the plot

plt.show()
from sklearn.metrics import confusion_matrix

results = model.predict_classes(X_test)

cm = confusion_matrix(np.where(y_test == 1)[1], results)

plt.matshow(cm)

plt.title('Confusion Matrix')

plt.colorbar()

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.show()
x_test.shape
m_samples = X_train.shape[0]

m_samplesTest = X_test.shape[0]

X_train1 = X_train.reshape(m_samples, -1)

X_test1 = X_test.reshape(m_samplesTest, -1)

 
print('Xtest shape',X_test1.shape)

print('X_train1 shape',X_train1.shape)

print('y_train shape',y_train.shape)

print('y_test shape',y_test.shape)





from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train1,y_train)



# Model Accuracy, how often is the classifier correct?

y_pred = clf.predict(X_test1)

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

print("Decision tree Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('F1 score : ',f1_score(y_test,y_pred,average="weighted"))

print('recall_score : ',recall_score(y_test,y_pred,average="weighted"))

print('precision_score : ',precision_score(y_test,y_pred,average="weighted"))
from xgboost import XGBClassifier

classifierG = XGBClassifier()

classifierG.fit(X_train1,y_train)

# Predicting the Test set results

y_predXG = classifierG.predict(X_test1)



from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

print("XGBOOST  Accuracy ON JAFFE data:",metrics.accuracy_score(y_test, y_pred))

print('F1 score : ',f1_score(y_test,y_pred,average="weighted"))

print('recall_score : ',recall_score(y_test,y_pred,average="weighted"))

print('precision_score : ',precision_score(y_test,y_pred,average="weighted"))
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train1,y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test1)

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

print("SVM  Accuracy ON JAFFE data:",metrics.accuracy_score(y_test, y_pred))

print('F1 score : ',f1_score(y_test,y_pred,average="weighted"))

print('recall_score : ',recall_score(y_test,y_pred,average="weighted"))

print('precision_score : ',precision_score(y_test,y_pred,average="weighted"))