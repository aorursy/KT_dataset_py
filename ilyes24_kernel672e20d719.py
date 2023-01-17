# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from tensorflow import keras

from PIL import Image

import os



dimen = 28



dir_path = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/'

sub_dir_list = os.listdir( dir_path )

images = list()

labels = list()

for i in range( len( sub_dir_list ) ):

    label = i

    image_names = os.listdir( dir_path + sub_dir_list[i] )

    for image_path in image_names:

        if image_path.startswith('C'):

            path = dir_path + sub_dir_list[i] + "/" + image_path

            image = Image.open( path ).convert( 'L' )

            resize_image = image.resize((dimen, dimen))

            array = list()

            for x in range(dimen):

                sub_array = list()

                for y in range(dimen):

                    sub_array.append(resize_image.load()[x, y])

                array.append(sub_array)

            image_data = np.array(array)

            image = np.array(np.reshape(image_data, (dimen, dimen, 1))) / 255

            images.append(image)

            labels.append( label )



x = np.array( images )

y = np.array( keras.utils.to_categorical( np.array( labels) , num_classes=2 ) )



num = 3000

limit = 10000



test_features = x[ 0 : num ]

test_labels = y[ 0 : num ]

train_features = x[ num : limit ]

train_labels = y[ num : limit ]



np.save( '/kaggle/working/x28.npy' , train_features )

np.save( '/kaggle/working/y28.npy' , train_labels )

np.save( '/kaggle/working/test_x28.npy' , test_features )

np.save( '/kaggle/working/test_y28.npy' , test_labels )

from tensorflow.python.keras import models , optimizers , losses ,activations

from tensorflow.python.keras.layers import *

from PIL import Image

import tensorflow as tf

import time

import os

import numpy as np



tf.compat.v1.disable_eager_execution()



class Classifier (object) :



    def __init__( self , number_of_classes ):

        

        dropout_rate = 0.2

        self.__DIMEN = 28



        input_shape = ( self.__DIMEN**2 , )

        convolution_shape = ( self.__DIMEN , self.__DIMEN , 1 )

        kernel_size = ( 3 , 3 )

        pool_size = ( 2 , 2 )

        strides = 1



        activation_func = activations.relu



        self.__NEURAL_SCHEMA = [



            Reshape( input_shape=input_shape , target_shape=convolution_shape),



            Conv2D( 16, kernel_size=kernel_size , strides=strides , activation=activation_func),

            Conv2D( 16, kernel_size=kernel_size , strides=strides , activation=activation_func),

            MaxPooling2D(pool_size=pool_size, strides=strides ),



            Conv2D( 32, kernel_size=kernel_size , strides=strides , activation=activation_func),

            Conv2D( 32, kernel_size=kernel_size , strides=strides , activation=activation_func),

            MaxPooling2D(pool_size=pool_size, strides=strides),



            Flatten(),



            Dense( 100, activation=activation_func) ,

            Dropout(dropout_rate),



            Dense( 100, activation=activation_func),

            Dropout(dropout_rate),



            Dense( number_of_classes, activation=tf.nn.softmax )



        ]



        self.__model = tf.keras.Sequential( self.__NEURAL_SCHEMA )



        self.__model.compile(

            optimizer=optimizers.Adam(),

            loss=losses.categorical_crossentropy ,

            metrics=[ 'accuracy' ] ,

        )



    def fit(self, X, Y  , hyperparameters):

        self.__model.fit(X, Y ,

                         batch_size=hyperparameters[ 'batch_size' ] ,

                         epochs=hyperparameters[ 'epochs' ] ,

                         callbacks=hyperparameters[ 'callbacks' ] ,

                         validation_data=hyperparameters[ 'val_data' ]

                         )

        self.__model.summary( )



    def prepare_images_from_dir( self , dir_path ) :

        images = list()

        images_names = os.listdir( dir_path )

        for imageName in images_names :

            print( imageName )

            image = Image.open(dir_path + imageName).convert('L')

            resize_image = image.resize((self.__DIMEN, self.__DIMEN))

            array = list()

            for x in range(self.__DIMEN):

                sub_array = list()

                for y in range(self.__DIMEN):

                    sub_array.append(resize_image.load()[x, y])

                array.append(sub_array)

            image_data = np.array(array)

            image = np.array(np.reshape(image_data,(self.__DIMEN, self.__DIMEN, 1)))

            images.append(image)



        return np.array( images )



    def evaluate(self , test_X , test_Y  ) :

        return self.__model.evaluate(test_X, test_Y)



    def predict(self, X  ):

        predictions = self.__model.predict( X  )

        return predictions



    def save_model(self , file_path ):

        self.__model.save('/kaggle/working/' + file_path )



    def load_model(self , file_path ):

        self.__model = models.load_model('/kaggle/working/' + file_path)



import numpy as np



data_dimension = 28



X = np.load( '/kaggle/working/x28.npy'.format( data_dimension ))

Y = np.load( '/kaggle/working/y28.npy'.format( data_dimension ))

test_X = np.load( '/kaggle/working/test_x28.npy'.format( data_dimension ))

test_Y = np.load( '/kaggle/working/test_y28.npy'.format( data_dimension ))



X = X.reshape( ( X.shape[0] , data_dimension**2  ) ).astype( np.float32 )

test_X = test_X.reshape( ( test_X.shape[0] , data_dimension**2 ) ).astype( np.float32 )



classifier = Classifier( number_of_classes=2 )

#classifier.load_model('0001.h5')



parameters = {

    'batch_size' : 250 ,

    'epochs' : 1 ,

    'callbacks' : None ,

    'val_data' : None

}



classifier.fit( X , Y  , hyperparameters=parameters )

#classifier.save_model( '0001.h5')



loss , accuracy = classifier.evaluate( test_X , test_Y )

print( "Loss of {}".format( loss ) , "Accuracy of {} %".format( accuracy * 100 ) )

print ( classifier.predict( test_X ).argmax( axis=1 ) )