# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install -U pandas_bokeh
import cv2
import gc
import os

import numpy as np 
import pandas as pd 

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
# import tensorflow.keras.applications.ResNet101 as resnet101

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
%matplotlib inline

# import pandas_bokeh
# from bokeh.models.widgets import DataTable, TableColumn
# from bokeh.models import ColumnDataSource

from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf

py.offline.init_notebook_mode(connected = True)
cf.go_offline()
cf.set_config_file(theme = 'solar')

# pd.set_option('plotting.backend', 'pandas_bokeh')
# pandas_bokeh.output_notebook()
print(tf.__version__)
SEED = 42
EPOCH = 20
BATCH_SIZE = 8
IMG_SIZE = 512
np.random.seed(SEED)
tf.random.set_seed(SEED)
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv',na_values=['unknown'])
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

train.head()
gc.collect()
DIR = '../input/siim-isic-melanoma-classification/jpeg/train/'
img = []
labels = []
jpg = '.jpg'

for i in train['image_name']:
    img.append(os.path.join(DIR,i)+jpg)
    
for i in train['target']:
    labels.append(i)
    
x_train,x_val,y_train,y_val = train_test_split(img,labels,test_size = 0.2,random_state = SEED)
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                         rescale=1.255,
                                        rotation_range=40,
                                         horizontal_flip=True,
                                         vertical_flip= True,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        
)

val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                            rescale=1./255
)
train_img = pd.DataFrame(x_train,columns=['image'])
train_labels = pd.DataFrame(y_train,columns=['target'])
train_data = pd.concat([train_img,train_labels],axis = 1)

val_img = pd.DataFrame(x_val,columns=['image'])
val_labels = pd.DataFrame(y_val,columns=['target'])
val_data = pd.concat([val_img,val_labels],axis = 1)

train_data.head()

train_img_gen = train_data_generator.flow_from_dataframe(train_data,
    x_col='image',
    y_col='target',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='raw')

val_img_gen = val_data_generator.flow_from_dataframe(val_data,
                                                    x_col = 'image',
                                                    y_col = 'target',
                                                    target_size= (IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    class_mode='raw')

# # , because of class imbalance it's better to use focal loss rather than normal binary_crossentropy.You can read more about it here

# def focal_loss(alpha=0.25,gamma=2.0):
#     def focal_crossentropy(y_true, y_pred):
#         bce = K.binary_crossentropy(y_true, y_pred)
        
#         y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
#         p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
#         alpha_factor = 1
#         modulating_factor = 1

#         alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
#         modulating_factor = K.pow((1-p_t), gamma)

#         # compute the final loss and return
#         return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
#     return focal_crossentropy
from tensorflow.python.keras import backend as K

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
from tensorflow.keras import layers, models

model = models.Sequential()
model.add(tf.keras.applications.ResNet101(weights='imagenet',
                                        include_top=False,
                                        input_shape=(IMG_SIZE,IMG_SIZE,3)
                                       ))
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(1024,activation= 'relu'))
model.add(layers.BatchNormalization())
# model.add(layers.Dense(70000,activation= 'relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(20000,activation= 'relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dense(1000,activation= 'relu'))
# model.add(layers.BatchNormalization())
model.add(layers.Dense(256,activation= 'relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1,activation='softmax'))
model.layers[0].trainable = False

model.compile(loss=focal_loss(),metrics=[tf.keras.metrics.AUC()],optimizer='adam' )
model.summary()
# from tf.keras.callbacks import ModelCheckpoint
filepath = "../working/saved_models-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'max')
callbacks_list = [checkpoint]
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_data.target),
                                                 train_data.target)
class_weight ={
    0:0.50893029,
    1:28.49462366
}
print(class_weight)
gc.collect()

History = model.fit_generator(train_img_gen,
                             steps_per_epoch=train_data.shape[0]//BATCH_SIZE,
                             epochs=EPOCH,
                             validation_data=val_img_gen,
                             validation_steps=val_data.shape[0]//BATCH_SIZE,
                             class_weight=class_weight,
                            callbacks=callbacks_list
                             )



np.unique(train_data.target)
tr