import tensorflow as tf
tf.__version__
from IPython.display import SVG

import tensorflow.keras as keras

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.utils.vis_utils import model_to_dot
model = Sequential()
model.add(Dense(32, input_shape=(224,)))

model.add(Activation('relu'))
model.summary()
model.output
SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.models import Model

from keras.layers import Input, Dense, concatenate, Reshape

from keras.applications.resnet50 import ResNet50
img_w = 224

img_h = 224
#rnet = ResNet50() # include_top=False,
#rnet.summary()
#print(rnet.output)

#print(rnet.outputs)
#x = rnet(a)

#x = Dense(1,input_shape=(1000,))(rnet.output)

#model = Model(inputs=rnet.input,outputs=x)
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#model.layers
rnet1 = ResNet50() # include_top=False,

rnet2 = ResNet50() # include_top=False,
rnet1.layers
rnet1.name = 'rnet1'

rnet2.name = 'rnet2'
i1 = Input(shape=(img_w,img_h,3))

i2 = Input(shape=(img_w,img_h,3))

#x = concatenate([i1,i2],axis=1)

#print(x.shape)
x1 = rnet1(i1)

x2 = rnet2(i2)
x1.shape, x2.shape
x = concatenate([x1,x2],axis=1)

x = Dense(50)(x) #input_shape=(2000,)

x = Dense(1)(x)

model = Model(inputs=[i1,i2],outputs=x)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
model.input_shape
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from keras import optimizers
def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
import pandas as pd

import os
!ls ../input/
data = pd.read_csv('../input/faces-train/all_data.csv',names=['im1','im2','label'],header=1)
data.shape
data['im1_good'] = data['im1'].apply(lambda x: x[1:] if x.startswith(' ') else x).apply(lambda x: os.path.join('../input/recognizing-faces-in-the-wild',x))

data['im2_good'] = data['im2'].apply(lambda x: x[1:] if x.startswith(' ') else x).apply(lambda x: os.path.join('../input/recognizing-faces-in-the-wild',x))
data.head()
import numpy as np
from matplotlib.pyplot import imread, imshow
N = 1000

subset = slice(N,2*N)
im1 = np.array([imread(x) for x in data.im1_good[subset]])

im2 = np.array([imread(x) for x in data.im2_good[subset]])
im1.shape,im2.shape
from keras.utils import to_categorical
x_train = [im1, im2]

y_train = data.label[subset]

np.shape(x_train), y_train.shape
y_train_res = np.array(y_train)

y_train_res.shape
# ahahahah... illuso

# h = model.fit(x_train, y_train_res, validation_split=0.2, epochs=4, batch_size=32)
# model.save('')