import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

import random

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.utils import np_utils

%matplotlib inline

from keras.models import Sequential

from keras.layers import Dense

from keras.regularizers import L1L2

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping

import tensorflow as tf

from keras.models import load_model

from tqdm import tqdm
# defining the path and classes.

directory = '../input/train'

test_directory = '../input/test/'

classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
def generate_path(*args,**kwargs):

    import os

    path_conv = ""

    for i in range(len(args)):

        

        if len(args) == 1 :

            path_conv = os.path.join(path_conv,args[i])     

            return path_conv

        elif i == (len(args) - 1):

            return path_conv

        

        path_conv = os.path.join(path_conv,args[i],args[i+1])
# provide the path and number of images to be displayed.

# function plots those images.

def display_images(path,no_of_images):

    count = 1

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array, cmap='gray')

        plt.show()

        count += 1

        if(no_of_images < count):

            break
# defining a shape to be used for our models.

img_size1 = 240

img_size2 = 240
class train_and_test:

    def __init__(self,*args,**kwargs):

        self.train_and_test = args

        

    # creating a training dataset.

    def create_training_data(self,path,classes,img_size1,img_size2):

            training_data = []

            for img in tqdm(os.listdir(path)):

                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_img = cv2.resize(img_array,(img_size2,img_size1))

                if classes == 'c0':

                    training_data.append([new_img,0])

                elif classes == 'c1' :               

                    training_data.append([new_img,1])

                elif classes == 'c2' :               

                    training_data.append([new_img,1])

                elif classes == 'c3' :               

                    training_data.append([new_img,1])

                elif classes == 'c4' :               

                    training_data.append([new_img,1])

                elif classes == 'c5' :               

                    training_data.append([new_img,1])

                elif classes == 'c6' :               

                    training_data.append([new_img,1])

                elif classes == 'c7' :               

                    training_data.append([new_img,1])

                elif classes == 'c8' :               

                    training_data.append([new_img,1])

                elif classes == 'c9' :               

                    training_data.append([new_img,1])

            return training_data

        

    # Creating a test dataset.    

    def create_testing_data(self,path,img_size1,img_size2):

        testing_data = []       

        for img in tqdm(os.listdir(path)):

            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

            new_img = cv2.resize(img_array,(img_size2,img_size1))

            testing_data.append([img,new_img])

        return testing_data
# Initializing the train and test classes for training and validation.

_train_ = train_and_test()

_test_ = train_and_test()
training_data_c0 = _train_.create_training_data(generate_path(directory,classes[0]),classes[0],img_size1,img_size2)

training_data_c1 = _train_.create_training_data(generate_path(directory,classes[1]),classes[1],img_size1,img_size2)

training_data_c2 = _train_.create_training_data(generate_path(directory,classes[2]),classes[2],img_size1,img_size2)

training_data_c3 = _train_.create_training_data(generate_path(directory,classes[3]),classes[3],img_size1,img_size2)

training_data_c4 = _train_.create_training_data(generate_path(directory,classes[4]),classes[4],img_size1,img_size2)

training_data_c5 = _train_.create_training_data(generate_path(directory,classes[5]),classes[5],img_size1,img_size2)

training_data_c6 = _train_.create_training_data(generate_path(directory,classes[6]),classes[6],img_size1,img_size2)

training_data_c7 = _train_.create_training_data(generate_path(directory,classes[7]),classes[7],img_size1,img_size2)

training_data_c8 = _train_.create_training_data(generate_path(directory,classes[8]),classes[8],img_size1,img_size2)

training_data_c9 = _train_.create_training_data(generate_path(directory,classes[9]),classes[9],img_size1,img_size2)
test_data = _test_.create_testing_data(generate_path(test_directory),img_size1,img_size2)
# create train and test data for our model.

class features_and_labels:

    # get all the arguments dynmically.

    def __init__(self,*args,**kwargs):

        self.features_and_labels = args

        

    # generate your features and labels.

    def generate_features_and_label(self,_class1_,_class2_):

        x = []

        y = []

        

        for features, label in tqdm(_class1_):

            x.append(features)

            y.append(label)

            

        for features, label in tqdm(_class2_):

            x.append(features)

            y.append(label)

            

        return x,y

    

    # generate np_arrays for test.

    def generate_npArray(self,_class1_,_class2_,img_size2,img_size1) :

        x,y = self.generate_features_and_label(_class1_,_class2_)

        np_array = np.array(x).reshape(-1,img_size2*img_size1)

        return np_array, y

    

    # train and split your data.

    def train_and_split(self,features,labels,test_size,random_state,num_class):

        x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=test_size,random_state=random_state)

        Y_train = np_utils.to_categorical(y_train,num_classes=num_class)

        Y_test = np_utils.to_categorical(y_test,num_classes=num_class)

        

        return x_train,x_test,Y_train,Y_test

    
feature_label = features_and_labels()



np_array_c0c1,y_c0c1 = feature_label.generate_npArray(training_data_c0,training_data_c1,img_size2,img_size1)

x_train_c0c1,x_test_c0c1,y_train_c0c1,y_test_c0c1 = feature_label.train_and_split(np_array_c0c1,y_c0c1,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 100

model_c0c1 = Sequential() 

model_c0c1.add(BatchNormalization())

model_c0c1.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c1.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c1 = model_c0c1.fit(x_train_c0c1, y_train_c0c1, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c1, y_test_c0c1),callbacks=callbacks) 
model_c0c1.save_weights('./driverdistraction_Safe_vs_texting_right_weights.h5', overwrite=True)

model_c0c1.save('./driverdistraction_lr_Safe_vs_texting_right.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c1.history['acc'])

plt.plot(history_c0c1.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c1.history['loss'])

plt.plot(history_c0c1.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c2, y_c0c2 = feature_label.generate_npArray(training_data_c0,training_data_c2,img_size2,img_size1)

x_train_c0c2,x_test_c0c2,y_train_c0c2,y_test_c0c2 = feature_label.train_and_split(np_array_c0c2,y_c0c2,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 25

model_c0c2 = Sequential() 

model_c0c2.add(BatchNormalization())

model_c0c2.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c2.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c2 = model_c0c2.fit(x_train_c0c2, y_train_c0c2, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c2, y_test_c0c2),callbacks=callbacks) 
model_c0c2.save_weights('./driverdistraction_talking_on_the_phone_right_weights.h5', overwrite=True)

model_c0c2.save('./driverdistraction_lr_talking_on_the_phone_right.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c2.history['acc'])

plt.plot(history_c0c2.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c2.history['loss'])

plt.plot(history_c0c2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c3, y_c0c3 = feature_label.generate_npArray(training_data_c0,training_data_c3,img_size2,img_size1)

x_train_c0c3,x_test_c0c3,y_train_c0c3,y_test_c0c3 = feature_label.train_and_split(np_array_c0c3,y_c0c3,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 25

model_c0c3 = Sequential() 

model_c0c3.add(BatchNormalization())

model_c0c3.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c3.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c3 = model_c0c3.fit(x_train_c0c3, y_train_c0c3, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c3, y_test_c0c3),callbacks=callbacks) 
model_c0c3.save_weights('./driverdistraction_Safe_texting_left_weights.h5', overwrite=True)

model_c0c3.save('./driverdistraction_lr_Safe_texting_left.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c3.history['acc'])

plt.plot(history_c0c3.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c3.history['loss'])

plt.plot(history_c0c3.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c4, y_c0c4 = feature_label.generate_npArray(training_data_c0,training_data_c4,img_size2,img_size1)

x_train_c0c4,x_test_c0c4,y_train_c0c4,y_test_c0c4 = feature_label.train_and_split(np_array_c0c4,y_c0c4,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 25

model_c0c4 = Sequential() 

model_c0c4.add(BatchNormalization())

model_c0c4.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c4.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c4 = model_c0c4.fit(x_train_c0c4, y_train_c0c4, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c4, y_test_c0c4),callbacks=callbacks) 
model_c0c4.save_weights('./driverdistraction_talking_on_the_phone_left_weights.h5', overwrite=True)

model_c0c4.save('./driverdistraction_lr_talking_on_the_phone_left.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c4.history['acc'])

plt.plot(history_c0c4.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c4.history['loss'])

plt.plot(history_c0c4.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c5, y_c0c5 = feature_label.generate_npArray(training_data_c0,training_data_c5,img_size2,img_size1)

x_train_c0c5,x_test_c0c5,y_train_c0c5,y_test_c0c5 = feature_label.train_and_split(np_array_c0c5,y_c0c5,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 25

model_c0c5 = Sequential() 

model_c0c5.add(BatchNormalization())

model_c0c5.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c5.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c5 = model_c0c5.fit(x_train_c0c5, y_train_c0c5, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c5, y_test_c0c5),callbacks=callbacks) 
model_c0c5.save_weights('./driverdistraction_operating_the_radio_weights.h5', overwrite=True)

model_c0c5.save('./driverdistraction_lr_operating_the_radio.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c5.history['acc'])

plt.plot(history_c0c5.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c5.history['loss'])

plt.plot(history_c0c5.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c6, y_c0c6 = feature_label.generate_npArray(training_data_c0,training_data_c6,img_size2,img_size1)

x_train_c0c6,x_test_c0c6,y_train_c0c6,y_test_c0c6 = feature_label.train_and_split(np_array_c0c6,y_c0c6,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 30

model_c0c6 = Sequential() 

model_c0c6.add(BatchNormalization())

model_c0c6.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c6.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c6 = model_c0c6.fit(x_train_c0c6, y_train_c0c6, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c6, y_test_c0c6),callbacks=callbacks) 
model_c0c6.save_weights('./driverdistraction_drinking_weights.h5', overwrite=True)

model_c0c6.save('./driverdistraction_lr_drinking.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c6.history['acc'])

plt.plot(history_c0c6.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c6.history['loss'])

plt.plot(history_c0c6.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c7, y_c0c7 = feature_label.generate_npArray(training_data_c0,training_data_c7,img_size2,img_size1)

x_train_c0c7,x_test_c0c7,y_train_c0c7,y_test_c0c7 = feature_label.train_and_split(np_array_c0c7,y_c0c7,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 30

model_c0c7 = Sequential() 

model_c0c7.add(BatchNormalization())

model_c0c7.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c7.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c7 = model_c0c7.fit(x_train_c0c7, y_train_c0c7, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c7, y_test_c0c7),callbacks=callbacks) 
model_c0c7.save_weights('./driverdistraction_reach_behind_weights.h5', overwrite=True)

model_c0c7.save('./driverdistraction_lr_reach_behind.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c7.history['acc'])

plt.plot(history_c0c7.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c7.history['loss'])

plt.plot(history_c0c7.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c8, y_c0c8 = feature_label.generate_npArray(training_data_c0,training_data_c8,img_size2,img_size1)

x_train_c0c8,x_test_c0c8,y_train_c0c8,y_test_c0c8 = feature_label.train_and_split(np_array_c0c8,y_c0c8,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 30

model_c0c8 = Sequential() 

model_c0c8.add(BatchNormalization())

model_c0c8.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c8.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c8 = model_c0c8.fit(x_train_c0c8, y_train_c0c8, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c8, y_test_c0c8),callbacks=callbacks) 
model_c0c8.save_weights('./driverdistraction_hair_and_makeup_weights.h5', overwrite=True)

model_c0c8.save('./driverdistraction_lr_hair_and_makeup.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c8.history['acc'])

plt.plot(history_c0c8.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c8.history['loss'])

plt.plot(history_c0c8.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
np_array_c0c9, y_c0c9 = feature_label.generate_npArray(training_data_c0,training_data_c9,img_size2,img_size1)

x_train_c0c9,x_test_c0c9,y_train_c0c9,y_test_c0c9 = feature_label.train_and_split(np_array_c0c9,y_c0c9,0.3,100,2)
# initializing the logistic regression classifier.

output_dim = nb_classes = 2

batch_size = 128 

nb_epoch = 25

model_c0c9 = Sequential() 

model_c0c9.add(BatchNormalization())

model_c0c9.add(Dense(output_dim, input_dim=240*240, activation='softmax')) 

model_c0c9.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

callbacks = [EarlyStopping(monitor='val_acc',patience=5,mode='max')]

history_c0c9 = model_c0c9.fit(x_train_c0c9, y_train_c0c9, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(x_test_c0c9, y_test_c0c9),callbacks=callbacks) 
model_c0c9.save_weights('./driverdistraction_talking_to_the_passenger_weights.h5', overwrite=True)

model_c0c9.save('./driverdistraction_lr_talking_to_the_passenger.h5')
# Plot training & validation accuracy values

plt.plot(history_c0c9.history['acc'])

plt.plot(history_c0c9.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history_c0c9.history['loss'])

plt.plot(history_c0c9.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn.metrics import r2_score

test_data_ = np.array(test_data[3000][1]).reshape(-1,img_size2*img_size1)

new_img = cv2.resize(test_data[3000][1],(img_size2,img_size1))

plt.imshow(new_img,cmap='gray')

plt.show()

pred = model_c0c3.predict(test_data_)

#r2_score(y_test, pred)

print(np.argmax(pred))
pred