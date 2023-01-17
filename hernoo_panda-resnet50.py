

from tensorflow import keras

keras.__version__


import tensorflow as tf

import numpy as np



class QuadraticWeightedKappa(tf.keras.metrics.Metric):

    def __init__(self, maxClassesCount=6, name='Kappa', **kwargs):        

        super(QuadraticWeightedKappa, self).__init__(name=name, **kwargs)

        self.M = maxClassesCount



        self.O = self.add_weight(name='O', initializer='zeros',shape=(self.M,self.M,), dtype=tf.int64)

        self.W = self.add_weight(name='W', initializer='zeros',shape=(self.M,self.M,), dtype=tf.float32)

        self.actualHist = self.add_weight(name='actHist', initializer='zeros',shape=(self.M,), dtype=tf.int64)

        self.predictedHist = self.add_weight(name='predHist', initializer='zeros',shape=(self.M,), dtype=tf.int64)

        

        # filling up the content of W once

        w = np.zeros((self.M,self.M),dtype=np.float32)

        for i in range(0,self.M):

            for j in range(0,self.M):

                w[i,j] = (i-j)*(i-j) / ((self.M - 1)*(self.M - 1))

        self.W.assign(w)

    

    def reset_states(self):

        """Resets all of the metric state variables.

        This function is called between epochs/steps,

        when a metric is evaluated during training.

        """

        # value should be a Numpy array

        zeros1D = np.zeros(self.M)

        zeros2D = np.zeros((self.M,self.M))

        tf.keras.backend.batch_set_value([

            (self.O, zeros2D),

            (self.actualHist, zeros1D),

            (self.predictedHist,zeros1D)

        ])







    def update_state(self, y_true, y_pred, sample_weight=None):

        # shape is: Batch x 1

        y_true = tf.reshape(y_true, [-1])

        y_pred = tf.reshape(y_pred, [-1])



        y_true_int = tf.cast(tf.math.round(y_true), dtype=tf.int64)

        y_pred_int = tf.cast(tf.math.round(y_pred), dtype=tf.int64)



        confM = tf.math.confusion_matrix(y_true_int, y_pred_int, dtype=tf.int64, num_classes=self.M)



        # incremeting confusion matrix and standalone histograms

        self.O.assign_add(confM)



        cur_act_hist = tf.math.reduce_sum(confM, 0)

        self.actualHist.assign_add(cur_act_hist)



        cur_pred_hist = tf.math.reduce_sum(confM, 1)

        self.predictedHist.assign_add(cur_pred_hist)



    def result(self):

        EFloat = tf.cast(tf.tensordot(self.actualHist,self.predictedHist, axes=0),dtype=tf.float32)

        OFloat = tf.cast(self.O,dtype=tf.float32)

        

        # E must be normalized "such that E and O have the same sum"

        ENormalizedFloat = EFloat / tf.math.reduce_sum(EFloat) * tf.math.reduce_sum(OFloat)



        

        return 1.0 - tf.math.reduce_sum(tf.math.multiply(self.W, OFloat))/tf.math.reduce_sum(tf.multiply(self.W, ENormalizedFloat))




#Import modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from os import walk

import imageio

from sklearn.model_selection import train_test_split

import tensorflow as tf

import gc #garbage collection





sampling=8900

height=512

width=512

classes = {0 : "0", 1 : "1", 2 : "2", 3 : "3", 4 : "4", 5 : "5"} #5 grades ISUP

batch_size = 0

#Standard preprocessing 

#load csv

#base_path='../input/explo-and-pre-processing'

#train=pd.read_csv(f'{base_path}/train.csv') # /!\ train is a bad name, it has training and validation in it. 



# just to be sure we will call every image contained in the folder

#(_, _, list_of_filenames) = next(walk(f"{base_path}/processed_pictures_rotation_included/"))

#list_of_filenames = [sub.replace('.tiff', '') for sub in list_of_filenames] 

#train=train[train['image_id'].isin(list_of_filenames)]



# Here we need it to feed the tensorFlow "tensor" with the path to the images

#train['path_to_image'] = train.apply (lambda x: f"{base_path}/processed_pictures_rotation_included/{x['image_id']}.tiff", axis=1)



#final_validation=pd.read_csv(f'{base_path}/test.csv')

"""

#Tile 512 512

#load csv

base_path='../input/panda-tile-512x512'

train=pd.read_csv(f'../input/explo-and-pre-processing/train.csv') # /!\ train is a bad name, it has training and validation in it. 

final_test=pd.read_csv(f'../input/prostate-cancer-grade-assessment/test.csv')

"""

#pre processing homemade 

base_path='../input/tile-pre-processing/512x512x3/'

train=pd.read_csv(f'../input/prostate-cancer-grade-assessment/train.csv') # /!\ train is a bad name, it has training and validation in it. 

final_test=pd.read_csv(f'../input/prostate-cancer-grade-assessment/test.csv')



# just to be sure we will call every image contained in the folder

(_, _, list_of_filenames) = next(walk(f"{base_path}/"))

list_of_filenames = [sub.replace('.png', '') for sub in list_of_filenames] 

train=train[train['image_id'].isin(list_of_filenames)]

if sampling:

    train=train.sample(n=sampling)

X_train, X_test, y_train, y_test = train_test_split(

    train, train.isup_grade, test_size=0.33, random_state=42)





def decode_img(img):

  # convert the compressed string to a 3D uint8 tensor

  img = tf.image.decode_png(img, channels=3)

  # resize the image to the desired size

  return tf.image.resize(img, [height,width])



def process_path(file_path,label):

    label = tf.one_hot(label, len(classes))

    label = tf.reshape(label, (1,6))

    # load the raw data from the file as a string

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    img = img/255

    img=tf.reshape(img, [1,512,512,3])

#     img = img.reshape(-1,512,512,3)

    

    return img, label



def get_label(file_path):

    # convert the path to a list of path components

    parts = tf.strings.split(file_path, os.path.sep)

    # The second to last is the class-directory

    one_hot = parts[-2] == class_names

    # Integer encode the label

    return tf.argmax(one_hot)

    

list_ds = tf.data.Dataset.from_tensor_slices((base_path+X_train.image_id+'.png',X_train.isup_grade))

#list_ds = tf.data.Dataset.list_files(base_path+'*', shuffle=False)



for f,i in list_ds.take(5):

    print(f.numpy())

    print(i.numpy())

train_ds=list_ds

train_ds = train_ds.map(process_path)

for image, label in train_ds.take(1):

    print("Image : ", image)

    print("Image shape: ", image.numpy().shape)

    print("Label: ", label.numpy())
X_train
""" sauvegarde

    X_train['np_image'] = X_train.apply (lambda x : (imageio.imread(f"{base_path}{x['image_id']}.png")/255).reshape([-1,512,512,3]), axis=1)

    X_test['np_image'] = X_test.apply (lambda x : (imageio.imread(f"{base_path}{x['image_id']}.png")/255).reshape([-1,512,512,3]), axis=1)



    X_train_np_image_from_df=np.stack(X_train['np_image'])

    y_train=tf.one_hot(y_train, 6)

    y_train=tf.reshape(y_train, [len(X_train),1,6])

    del X_train

    #X_train_isup_from_df=np.array(X_train['isup_grade'])

    X_test_np_image_from_df=np.stack(X_test['np_image'])

    y_test=tf.one_hot(y_test, 6)

    y_test=tf.reshape(y_test, [len(X_test),1,6])

    del X_test

    gc.collect()

    #X_test_isup_from_df=np.array(X_test['isup_grade'])

















    #useful for TF classes

    

    # Here we will feed the tensorFlow database 



    training_dataset = (

        tf.data.Dataset.from_tensor_slices(

            (

                X_train_np_image_from_df,

                y_train

                )

        )

    )



    del(X_train_np_image_from_df)

    validation_dataset = (

        tf.data.Dataset.from_tensor_slices(

            (

                X_test_np_image_from_df,

                y_test

                )

        )

    )

    del(X_test_np_image_from_df)

    gc.collect()

    # garbage collection / deleting





    batch_size = 0

    width = 512

    height= 512

   

    # Output preview

    for features_tensor, target_tensor in training_dataset.take(5):

        print(f'features:{features_tensor} target:{target_tensor}')

        #The tensor has the image path, and the label coded.



    width = 512

    height= 512

    # 0 means that it won't be batched

    #the function takes the first and the second item of the tensorflow item

    #thanks to map it will be passed "row by row"

    def _parse_function(filename, label, h=height, w=width, rotating=True): 

        image = tf.io.read_file(filename) #reading the image in the memory

        #image = tfio.experimental.image.decode_tiff(image, index=0, name=None) # tiff to numpy 1/2

        image = tf.io.decode_png(image,dtype=tf.dtypes.uint8, name=None)

        if rotating==True :

            sh=tf.shape(tf.io.read_file(filename))

            print(f"{sh}")

            width, height = image.shape[0],image.shape[1]

            print(f"w={width}, h={height}")

            image = tf.image.convert_image_dtype(image, tf.float32) # converting it 2/2

    #     

        image = image/255 #normalisation

        #image = tf.image.resize(image, [h, w]) #resize

        image = tf.cast(image, tf.float32) # transforming into a tf.float object



        return image, label



    tr_dataset = training_dataset.map(_parse_function)

    va_dataset = validation_dataset.map(_parse_function)

    

    if batch_size != 0:

        training_dataset = training_dataset.batch(batch_size, drop_remainder=True) # drop_remainder is to drop the last batch which is not complete

        validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True)

"""
import sys

def sizeof_fmt(num, suffix='B'):

    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f %s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f %s%s" % (num, 'Yi', suffix)



for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),

                         key= lambda x: -x[1])[:10]:

    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
from keras.models import load_model

from sklearn.datasets import load_files   

from keras.utils import np_utils

from glob import glob

from keras import applications

from keras.preprocessing.image import ImageDataGenerator 

from keras import optimizers

from keras.models import Sequential,Model,load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint



base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (height,width,3))
num_classes= len(classes)

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.7)(x)

predictions = Dense(num_classes, activation= 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

from keras.optimizers import SGD, Adam

# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

adam = Adam(lr=0.0001)

model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy',QuadraticWeightedKappa()])
history=model.fit(train_ds, epochs = 55, batch_size = batch_size)
#history retrieving

print(history.history.keys())

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['Kappa'])

plt.plot(history.history['val_Kappa'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#save model

model.save('resnet50')

