# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#install efficientnet

!pip install -U efficientnet
import os

import time

import cv2

import keras

import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.optimizers import Adam, SGD

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical

from keras.utils import np_utils

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint

import albumentations as albu

from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

import efficientnet.keras as efn 
# efficient net has good performance for 224,224,3 images so let us define input_shape and other variables accordingly

input_shape = (224, 224, 3)

n_classes = 100 #total number categories of images

epochs = 25 

lr = 1e-3 #learning_rate = 0.001

batch_size = 8 #number of images to be selected per batch
#read data and split them into train and validation

df = pd.read_csv('/kaggle/input/cmpe258-lab1/CIFAR_train_images.csv')

df_labels = pd.read_csv('/kaggle/input/cmpe258-lab1/CIFAR_train_labels.csv')



X_train = df.values

X_train = X_train[:, 0:].reshape(X_train.shape[0], 32, 32, 3).astype('float32')

X_train = X_train / 255.0



Y_train = df_labels.values



df_test = pd.read_csv('/kaggle/input/cmpe258-lab1/CIFAR_test_images.csv')

X_test = df_test

X_test = X_test.drop(columns=['index'], axis=1)

X_test = X_test.values

X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')



x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=2)
#defining a function resize as CIFAR images are 32 x 32 x 3 we will need to resize them to 224 x 224 x 3

def img_resize(img, shape):

    return cv2.resize(img, (shape[1], shape[0]), interpolation = cv2.INTER_CUBIC)
#To generate Augmented data per batch for training over all the samples let us create a class DataGen with two modes fit and predict

class DataGen(keras.utils.Sequence):

    def __init__(self, images , labels = None, mode = 'fit', batch_size = batch_size,

                 dim = (224, 224), channels = 3, n_classes = n_classes,

                 shuffle = True, augment = False):

        self.augment = augment

        self.images = images

        self.n_classes = n_classes

        self.dim = dim

        self.labels = labels

        self.shuffle = shuffle

        self.mode = mode

        self.batch_size = batch_size

        self.channels = channels

        

        self.on_epoch_end()

     

                

    def __len__(self):

        return int(np.floor(len(self.images) / self.batch_size))

    

    

    # generate augmented images per batch in real-time

    def _batchAugmentation(self, img_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i] = self._transform(img_batch[i])

            

        return img_batch

    

    # define augmentation parameters using albumenations                      

    def _transform(self, img):

        composition = albu.Compose([albu.HorizontalFlip(p = 0.5),

                                    albu.VerticalFlip(p = 0.5),

                                    albu.GridDistortion(p = 0.2),

                                    albu.ElasticTransform(p = 0.2)])

        

        return composition(image = img)['image']

    

    # let us define a function to get different image batches after each epoch        

    def on_epoch_end(self):

        self.indicies = np.arange(self.images.shape[0])

        if self.shuffle == True:

            np.random.shuffle(self.indicies)

            

    # define a function to get the training batches            

    def __getitem__(self, index):

        batch_indicies = self.indicies[index*self.batch_size:(index+1)*self.batch_size]

        X = np.empty((self.batch_size, *self.dim, self.channels))

        for i, ID in enumerate(batch_indicies):

            img = self.images[ID]

            img = img.astype(np.float32) / 255.

            img = img_resize(img, self.dim)

            X[i] = img

            

        if self.mode == 'fit':

            #get batches of labels and perform one-hot encoding

            y = self.labels[batch_indicies]

            y = to_categorical(y, n_classes)

    

            if self.augment == True:

                X = self._batchAugmentation(X)                

            

            return X,y

        

        elif self.mode == 'predict':

            return X       

        

        else:

            raise AttributeError('The mode parameters should be set to "fit" or "predict"')
#generate train and validation after preprocessing, one-hot encoding and  real-time augmentation

generate_trainData = DataGen(x_train, y_train, augment = True)

generate_valData = DataGen(x_val, y_val, augment = False)
#define constants such as decay rate, patience etc.,

decay_rate = 0.5

es_patience = 10

rlrop_patience = 5



#define efficient net parameter with imagenet weights without top 

efnb0 = efn.EfficientNetB0(weights = 'imagenet', include_top = False, classes = n_classes, input_shape = input_shape)



#define sequential model

def _createModel():

    model = Sequential()

    model.add(efnb0)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.2)) 

    model.add(Dense(n_classes, activation='softmax')) #FC1 with 100 nodes and softmax activation

    efnb0.trainable = True



    #monitor val_loss to schedule lr accordingly and to store best_weights

    rlrop = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', patience = rlrop_patience, 

                        factor = decay_rate, min_lr = 1e-6, verbose = 1)

    sgd = SGD(lr = lr, momentum = 0.9, nesterov = True)

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = es_patience, restore_best_weights = True, verbose = 1)

    

    return model
# This function has three modes 'mode 0: _performTraining', 'mode 1: _trainStartFrom_BestWeights and train as per observations' and 'mode 2: _makePredictions_BestModel'

def _evaluation(mode):

    if mode == 0:

        model = _createModel()

        #comppile the model

        model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

        model_type = 'EfficientNetB0'



        #create the directory where to store the model and to name the models accordingly

        save_dir = os.path.join(os.getcwd(), 'saved_CIFARv1_models')

        model_name = 'cifar100_%s_model.{epoch:03d}.h5' % model_type

        if not os.path.isdir(save_dir):

            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, model_name)



        # create checkpoints to monitor and store the model based on val_accuracy

        checkpoint = ModelCheckpoint(filepath=filepath,

                             monitor='val_accuracy',

                             verbose=1,

                             save_best_only=True)

        #use callbacks to callback monitored parameters to store weights and models

        callbacks = [checkpoint, es, rlrop]



        #display total training time for CIFAR100

        t = time.time()

        #fit the model using keras fit_generator

        hist = model.fit_generator(generate_trainData,validation_data = generate_valData, 

                           epochs = epochs, verbose = 1, callbacks = callbacks)

        print('Training time: %s' % (t - time.time()))

        #preprocess the test data using predict mode and keeping augmentation and shuffle false

        test_generator = DataGenerator(x_val, mode = 'predict', augment = False, shuffle = False)

        y_pred = model.predict_generator(test_generator,verbose = 1)

        y_pred = np.argmax(y_pred, axis = 1)

        d = {'index': df_test['index'].values, 'answer': y_pred}

        res_df = pd.DataFrame(data=d)

        res_df.to_csv('CIFAR100EfficientNetB0.csv', index=False)

        print("The accuracy on the testing data : {:.2f}%".format(100 * accuracy_score(y_test, y_pred)))

        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred))) 

        

        #Predict and write to results on the given test data to csv file

        test_generator = DataGenerator(X_test, mode = 'predict', augment = False, shuffle = False)

        Y_pred = model.predict_generator(test_generator,verbose = 1)

        Y_pred = np.argmax(y_pred, axis = 1)

        d = {'index': df_test['index'].values, 'answer': Y_pred}

        res_df = pd.DataFrame(data=d)

        res_df.to_csv('CIFAR100EfficientNetB0.csv', index=False)

        

        return model.summary()

        

    elif mode == 1:

        model = _createModel()

        # to save time on training

        

        #load the weights file in here.

        weights_path = '/kaggle/input/weights-effnet/bestWeight.h5' 

        

        

        model.load_weights()

        #comppile the model

        model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

        model_type = 'EfficientNetB0'



        #create the directory where to store the model and to name the models accordingly

        save_dir = os.path.join(os.getcwd(), 'saved_CIFARv1_models')

        model_name = 'cifar100_%s_model.{epoch:03d}.h5' % model_type

        if not os.path.isdir(save_dir):

            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, model_name)



        # create checkpoints to monitor and store the model based on val_accuracy

        checkpoint = ModelCheckpoint(filepath=filepath,

                             monitor='val_accuracy',

                             verbose=1,

                             save_best_only=True)

        #use callbacks to callback monitored parameters to store weights and models

        callbacks = [checkpoint, es, rlrop]



        #display total training time for CIFAR100

        t = time.time()

        #fit the model using keras fit_generator

        hist = model.fit_generator(generate_trainData,validation_data = generate_valData, 

                           epochs = 5, verbose = 1, callbacks = callbacks)

        print('Training time: %s' % (t - time.time()))



        #preprocess the test data using predict mode and keeping augmentation and shuffle false

        test_generator = DataGenerator(x_val, mode = 'predict', augment = False, shuffle = False)

        y_pred = model.predict_generator(test_generator,verbose = 1)

        y_pred = np.argmax(y_pred, axis = 1)

        print("The accuracy on the testing data : {:.2f}%".format(100 * accuracy_score(y_val, y_pred)))

        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        

        #Predict and write to results on the given test data to csv file

        test_generator = DataGenerator(X_test, mode = 'predict', augment = False, shuffle = False)

        Y_pred = model.predict_generator(test_generator,verbose = 1)

        Y_pred = np.argmax(y_pred, axis = 1)

        d = {'index': df_test['index'].values, 'answer': Y_pred}

        res_df = pd.DataFrame(data=d)

        res_df.to_csv('CIFAR100EfficientNetB0.csv', index=False)

        

        return model.summary()





    elif mode == 2:

        

        #define path here of the best saved model

        model_path = '/saved_CIFARv1_models/cifar100_EfficientNetB0_model.025.h5'

        

        model = load_model(model_path)

        #preprocess the test data using predict mode and keeping augmentation and shuffle false

        test_generator = DataGenerator(x_val, mode = 'predict', augment = False, shuffle = False)

        y_pred = model.predict_generator(test_generator,verbose = 1)

        y_pred = np.argmax(y_pred, axis = 1)

        d = {'index': df_test['index'].values, 'answer': y_pred}

        res_df = pd.DataFrame(data=d)

        res_df.to_csv('CIFAR100EfficientNetB0.csv', index=False)

        print("The accuracy on the testing data : {:.2f}%".format(100 * accuracy_score(y_val, y_pred)))

        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        

        #Predict and write to results on the given test data to csv file

        test_generator = DataGenerator(X_test, mode = 'predict', augment = False, shuffle = False)

        Y_pred = model.predict_generator(test_generator,verbose = 1)

        Y_pred = np.argmax(y_pred, axis = 1)

        d = {'index': df_test['index'].values, 'answer': Y_pred}

        res_df = pd.DataFrame(data=d)

        res_df.to_csv('CIFAR100EfficientNetB0.csv', index=False)

                

        return model.summary()

                     

    else:  

        raise AttributeError('The mode parameters should be set to 0 -> _performTraining or 1 -> _startTrainingFrom_BestWeights or 2 -> _makePredictionsFrom_BestModel')
_evaluation(0)