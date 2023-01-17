# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt

import cv2
%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_DETg9GD/train.csv')
train.head()

import cv2
Dir = '../input/train_DETg9GD/Train'
imgs = []
for i,p in enumerate(train['ID']):
    path = os.path.join(Dir,p)
    if not i%1000:
        print(i,path)
    img = cv2.imread(path)
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img.astype('float32')
    imgs.append(img)
X = np.stack(imgs)
X = X.reshape(X.shape[0], 64, 64, 1)
X = X.astype('float32')
X = X/255
np.save('X.npy',X)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
lb = LabelEncoder()

y = lb.fit_transform(train['Class'])
y = np_utils.to_categorical(y)

np.save('y.npy',y)
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.25)
batch_size = 64
epochs = 50
MODEL_NAME = 'age_convnet'




def build_model():
    
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(64,64,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

# Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax') )
    
    return model
def train(model, X_train, Y_train, X_test, Y_test):
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    model.fit(X_train, Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, Y_test))
def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    
    

    model = build_model()

    train(model, X_train, Y_train, X_test, Y_test)
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "activation_6/Softmax")
    
if __name__ == '__main__':
    main()







