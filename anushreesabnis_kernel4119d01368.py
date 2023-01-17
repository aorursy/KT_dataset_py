# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

#/kaggle/input/fer2013.csv



import numpy as np

import keras

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import keras.backend as K

K.set_image_data_format("channels_last")

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import math

import h5py

import tensorflow as tf

from tensorflow.keras import callbacks





%matplotlib inline

import pandas as pd



import cv2 





# Read Metadata

meta_df = pd.read_csv(r"/kaggle/input/emotion-detection/fer2013.csv")



meta_df.replace(to_replace ={'emotion': {1:0}}, inplace = True)



emotions = ['Angry','Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

classes = len(emotions)

meta_df = meta_df.rename(columns={"emotion":"label","pixels":"image"})

meta_df['emotion'] = meta_df['label'].apply(lambda x: emotions[int(x)])

meta_df = meta_df.drop(columns=['Usage'])



print(classes)
meta_df['emotion'] = meta_df['label'].apply(lambda x: emotions[int(x)])
meta_df
labels = []

for i in meta_df['emotion']:

    labels.append(i)

print(labels[388])
#labels = []

#for i in meta_df['label']:

#labels.append(i)

#print(labels[388])

#labels1 = []

#for id in range(35887):

#    if labels[id] == 1:

#        labels1.append(id)

#print(labels1[1])

#meta_df['label'] = meta_df['label'].replace({1: 0})
print(labels[388])
print(classes)
train_set_x_orig=[]

for id in range(35887):

    image = np.reshape(np.array(meta_df.image[id].split(' ')).astype(int),(48,48, 1))

    train_set_x_orig.append(image)

train_set_x_orig = np.array(train_set_x_orig)

print(train_set_x_orig.shape)

train_set_y_orig = np.array(meta_df.label)

print(train_set_y_orig.shape)

test_set_x_orig = []

test_set_y_orig = []

test_set_x_orig = np.array(test_set_x_orig)

test_set_y_orig = np.array(test_set_y_orig)

from sklearn.model_selection import train_test_split

train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig = train_test_split(train_set_x_orig, train_set_y_orig,shuffle = True, stratify = train_set_y_orig, random_state = 2020, test_size = 0.2 )



print(train_set_x_orig.shape)

print(test_set_x_orig.shape)

print(train_set_y_orig.shape)

print(test_set_y_orig.shape)
val_set_x_orig = []

val_set_y_orig = []

val_set_x_orig = np.array(val_set_x_orig)

val_set_y_orig = np.array(val_set_y_orig)

from sklearn.model_selection import train_test_split

train_set_x_orig, val_set_x_orig, train_set_y_orig, val_set_y_orig = train_test_split(train_set_x_orig, train_set_y_orig,shuffle = True, stratify = train_set_y_orig, random_state = 2020, test_size = 0.2 )



print(train_set_x_orig.shape)

print(val_set_x_orig.shape)

print(val_set_y_orig.shape)

print(train_set_y_orig.shape)
# Normalize image vectors

X_train = train_set_x_orig/255.

X_test = test_set_x_orig/255.

X_val = val_set_x_orig/255.



# Reshape

Y_train = train_set_y_orig.T

Y_test = test_set_y_orig.T

Y_val = val_set_y_orig.T





print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("number of val examples = " + str(X_val.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_val shape: " + str(X_val.shape))

print ("Y_val shape: " + str(Y_val.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))
from keras.utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = classes)

Y_test = to_categorical(Y_test, num_classes = classes)

Y_val = to_categorical(Y_val, num_classes = classes)

print(Y_train.shape)

print(Y_val.shape)

print(Y_test.shape)
from keras.constraints import maxnorm
def Moodel1(input_shape):

   

    

    X_input = Input(input_shape)



    

    #Block 1 --padding -- conv2d 

    

    X = ZeroPadding2D((3,3))(X_input)



    X = Conv2D( 16, (3,3), strides = (1,1), padding = 'valid', name = 'conv0',kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    X = Activation('relu')(X)



    X = layers.Dropout(0.2)(X)





    X = Conv2D( 16, (3,3), strides = (1,1), padding = 'valid', name = 'conv1', kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)



    X = MaxPooling2D((3,3), strides = (1,1), name = 'max_pool_1')(X)



    X = Conv2D( 32, (3,3), strides = (1,1), padding = 'valid', name = 'conv2', kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn2')(X)

    X = Activation('relu')(X)



    X = MaxPooling2D(( 3,3 ), strides = (1,1), name = 'max_pool_2')(X)



    X = Conv2D( 32, (3,3), strides = (1,1), padding = 'valid', name = 'conv3' , kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn3')(X)

    X = Activation('relu')(X)



    X = MaxPooling2D(( 5,5 ), strides = (1,1), name = 'max_pool_3')(X)



    X = Conv2D( 64, (3,3), strides = (1,1), padding = 'valid', name = 'conv4' , kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn4')(X)

    X = Activation('relu')(X)



    X = MaxPooling2D(( 3,3 ), strides = (2,2), name = 'max_pool_4')(X)



    X = Conv2D( 128, (3,3), strides = (1,1), padding = 'same', name = 'conv5' , kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn5')(X)

    X = Activation('relu')(X)



    X = Conv2D(180 , (1,1), strides = (1,1), padding = 'valid', name = 'conv6' , kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn6')(X)

    X = Activation('relu')(X)



    X = layers.Dropout(0.2)(X)



    X = Conv2D( 256, (5,5), strides = (1,1), padding = 'valid', name = 'conv7' , kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn7')(X)

    X = Activation('relu')(X)



    X = MaxPooling2D(( 6,6 ), strides = (1,1), name = 'max_pool_7' )(X)



    X = Conv2D(512 , (3,3), strides = (1,1), padding = 'valid', name = 'conv8' , kernel_constraint = maxnorm(3))(X)

    X = BatchNormalization(axis = 3, name = 'bn8')(X)

    X = Activation('relu')(X)





    X = Flatten()(X)



    #o/p shape : (9,9,320)



    X = Dense(64, activation= 'relu', name = 'fc0')(X)



    X = Dense(32, activation = 'relu', name = 'fc1')(X)



    X = Dense(16, activation = 'relu', name = 'fc2' )(X)



    X = layers.Dropout(0.5)(X)



    X = Dense(7, activation = 'softmax', name = 'fc3')(X)





    model = Model(inputs = X_input, output = X, name = 'Moodel')

    

    return model



my_callback = [tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 4)]


# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    moodel = Moodel1(X_train.shape[1:])

    moodel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    
history = moodel.fit(X_train, Y_train, epochs = 70, batch_size=200, verbose = 1, validation_data = (X_val, Y_val), callbacks = my_callback)
moodel.evaluate(X_test,Y_test)
moodel.save(r"/kaggle/working/Moodel2_1.ipynb")

y_proba = moodel.predict(X_test)

print(y_proba)



        



    

    
Y_pred = []

for i in y_proba:

    Y_pred.append(i)

    

Y_pred1 = []

for id in range(7178):

    (m,i) = max((v,i) for i,v in enumerate(Y_pred[id]))

    Y_pred1.append((m,i)) #(5, 2)

    
Y_pred2 = np.asarray(Y_pred1)

Y_pred3 = []

for id in range(7178):

    Y_pred3.append(Y_pred2[id][1])

print(Y_pred3[1])



    
Y_pred3 = np.asarray(Y_pred3)

print(Y_pred3.shape)
print(Y_pred3)

Y_pred4 = []

for i in Y_pred3:

    Y_pred4.append(int(i))

print(Y_pred4)

    
Y_pred3[Y_pred3 == 1] = 0

if Y_pred3.any() == [1.]:

    print('False')

else:

    print('Done')

                                      

        
Y_test1 = Y_test

Y_test1 = np.asarray(Y_test1)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import itertools

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



y_true = Y_test1

y_pred = Y_pred4

cm = confusion_matrix(y_true, y_pred)

labels = ['Angry', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

title='Confusion matrix'

print(cm)



plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title(title)

plt.colorbar()

tick_marks = np.arange(len(labels))

plt.xticks(tick_marks, labels, rotation=45)

plt.yticks(tick_marks, labels)

fmt = 'd'

thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

    plt.text(j, i, format(cm[i, j], fmt),

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()