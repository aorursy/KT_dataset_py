#!pip uninstall --yes tensorflow tf-nightly

#!pip install tensorflow==2.2-rc1
FRAMES = 7
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import PIL.Image

from io import BytesIO

import IPython.display

import numpy as np

import cv2



def showarray(imgs,w=None,h=None):

    from IPython.display import Image

    if w is None:

      w = imgs.shape[0]

    if h is None:

      h = imgs.shape[1]

    limg = imgs*(255,255,255)

    _,ret = cv2.imencode('.png', limg)

    #_,data = cv2.imencode('.png', limg)

    #cv2img = cv2.imdecode(data, 1)

    #cv2img2 = cv2.resize(cv2img,(w,h), interpolation = cv2.INTER_AREA)

    #cv2.imshow('',cv2img2) 

    img = Image(data=ret.tobytes())

    return IPython.display.display(img)
from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import numpy as np

img = mpimg.imread('../input/galton-board/images/i0001.png')

img2 = mpimg.imread('../input/galton-board/images/i0008.png')



img = img.reshape((img.shape[0],img.shape[1],1))

img2 = img2.reshape((img2.shape[0],img2.shape[1],1))

print(img.shape)

print(img.dtype)
showarray(img[60:415,140:375,:],400,400)

input_shape = img[60:415,140:375,0:1].shape
import tensorflow.keras

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.optimizers import Adadelta,Adam

from tensorflow.keras.layers import  GaussianNoise, Conv3D, ConvLSTM2D, Dropout,  MaxPooling2D, Flatten, Dense, GaussianNoise, BatchNormalization

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from tensorflow.keras.regularizers import l2
class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self,path,fr,to,frames=20,batch_size=1):

        self.path = path

        self.fr = fr

        self.to   = to

        self.batch_size = batch_size

        if (self.to-self.fr) % self.batch_size != 0:

            raise Exception(f"Invalid batch_size {batch_size} (self.to-self.fr={self.to-self.fr})")

        self.frames = frames

        #print(f"self.data.shape:{self.data.shape}")

    def __len__(self):

        return (self.to - self.fr) // self.batch_size

    def __getitem__(self, idx):

        #print(f"__getitem__ {index},self.data[index].shape: {self.data[index].shape}")

        i1r = None

        i2r = None

        for b in range(0,self.batch_size):

            for i in range(idx,idx+self.frames):

                k = b*self.frames+self.fr+i

                if k > self.to+self.frames:

                    raise Exception("Invalid index")

                img = mpimg.imread('../input/galton-board/images/i%04d.png' % (k))

                img = img.reshape((img.shape[0],img.shape[1],1))

                img2 = mpimg.imread('../input/galton-board/images/i%04d.png' % (k+1))

                img2 = img2.reshape((img.shape[0],img.shape[1],1))

                i1 = img[60:415,140:375,0:1]

                i2 = img2[60:415,140:375,0:1]

                i1 = i1.reshape((i1.shape[0],i1.shape[1],i1.shape[2]))

                i2 = i2.reshape((i2.shape[0],i2.shape[1],i2.shape[2]))

                if i1r is None:

                    i1r = np.zeros(shape=(self.batch_size,self.frames,i1.shape[0],i1.shape[1],i1.shape[2]))

                if i2r is None:

                    i2r = np.zeros(shape=(self.batch_size,self.frames,i2.shape[0],i2.shape[1],i2.shape[2]))



                i1r[b][i-idx] = i1

                i2r[b][i-idx] = i2

        

        return i1r,i2r
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



__author__ = "Jens E. Köhler"



import numpy as np

import tensorflow as tf





def ndarray_to_tfrecords(X, Y, file_path, verbose=True):

    """ Converts a Numpy array (or two Numpy arrays) into a TFRecords format file.

    Description:

        Convert input data provided as numpy.ndarray into TFRecords format file.

    Args:

        X : (numpy.ndarray) of rank N

            Numpy array of M training examples. Its dtype should be float32, float64 or int64.

            X gets reshaped into rank 2, where the first dimension denotes to m (the number of

            examples) and the rest to the dimensions of one example. The shape of one example

            is stored to feature 'x_shape'.

        Y : (numpy.ndarray) of rank N or None

            Numpy array of M labels. Its dtype should be float32, float64, or int64.

            None if there is no label array. Y gets also reshaped into rank 2, similiar to X.

            The shape of one label is stored to feature 'y_shape'

        file_path: (str) path and name of the resulting tfrecord file to be generated

        verbose : (bool) if true, progress is reported.

    Raises:

        ValueError: if input type is not float64, float32 or int64.

    """

    

    

    def _dtype_feature(ndarray):

        """match appropriate tf.train.Feature class with dtype of ndarray"""

        assert isinstance(ndarray, np.ndarray)

        dtype_ = ndarray.dtype

        if dtype_ == np.float64 or dtype_ == np.float32:

            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))

        elif dtype_ == np.int64:

            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))

        else:

            raise ValueError("The input should be numpy ndarray. Instead got {}".format(ndarray.dtype))

    

    assert isinstance(X, np.ndarray)

    X_flat = np.reshape(X, [X.shape[0], np.prod(X.shape[1:])])

    dtype_X = _dtype_feature(X_flat)

    

    assert isinstance(Y, np.ndarray) or Y is None

    if Y is not None:

        assert X.shape[0] == Y.shape[0]

        Y_flat = np.reshape(Y, [Y.shape[0], np.prod(Y.shape[1:])])

        dtype_Y = _dtype_feature(Y_flat)

    

    # Generate tfrecord writer

    with tf.io.TFRecordWriter(file_path) as writer:

        

        if verbose:

            print("Serializing {:d} examples into {}".format(X.shape[0], file_path))

        

        # iterate over each sample and serialize it as ProtoBuf.

        for idx in range(X_flat.shape[0]):

            if verbose:

                print("- write {0:d} of {1:d}".format(idx, X_flat.shape[0]), end="\r")

            

            x = X_flat[idx]

            x_sh = np.asarray(X.shape[1:])

            dtype_xsh = _dtype_feature(x_sh)

            

            if Y is not None:

                y = Y_flat[idx]

                y_sh = np.asarray(Y.shape[1:])

                dtype_ysh = _dtype_feature(y_sh)

            

            d_feature = {}

            d_feature["X"] = dtype_X(x)

            d_feature["x_shape"] = dtype_xsh(x_sh)

            if Y is not None:

                d_feature["Y"] = dtype_Y(y)

                d_feature["y_shape"] = dtype_ysh(y_sh)

            

            features = tf.train.Features(feature=d_feature)

            example = tf.train.Example(features=features)

            serialized = example.SerializeToString()

            writer.write(serialized)

    

    if verbose:

        print("Writing {} done!".format(file_path))
from os import path

val_split = 10

FROM1=1

TO1=608

FROM2=700

TO2=899

training_generator   = DataGenerator(path='../input/galton-board/images/',

                                   fr=FROM1,

                                   to=TO1,

                                   frames=FRAMES,

                                   batch_size=1)

validation_generator = DataGenerator(path='../input/galton-board/images/',

                                   fr=FROM2,

                                   to=TO2,

                                   frames=FRAMES,

                                   batch_size=1)

for i in range(0,len(training_generator)):

    if not path.isfile("/kaggle/working/%04d_train.tfrecord" % i):

        ndarray_to_tfrecords(training_generator[i][0],training_generator[i][1],"/kaggle/working/%04d_train.tfrecord" % i)

        if not path.isfile("/kaggle/working/%04d_train.tfrecord" % i):

            raise Exception("Failed creating: %04d_train.tfrecord" % i)

for i in range(0,len(validation_generator)):

    if not path.isfile("/kaggle/working/%04d_validate.tfrecord" % i):

        ndarray_to_tfrecords(validation_generator[i][0],validation_generator[i][1],"/kaggle/working/%04d_validate.tfrecord" % i)

        if not path.isfile("/kaggle/working/%04d_validate.tfrecord" % i):

            raise Exception("Failed creating: %04d_validate.tfrecord" % i)

!tar cfvz tfrecords.tar.gz *.tfrecord
!rm *.tfrecord
!ls -lah