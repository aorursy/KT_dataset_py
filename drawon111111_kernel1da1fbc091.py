# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras import layers,models

from keras.preprocessing.image import img_to_array

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')

import os,shutil

from sklearn.model_selection import train_test_split

from keras import applications

from keras.preprocessing.image import image

from keras import layers,models,optimizers

import math
train_images_dir="../input/aptos2019-blindness-detection/train_images"

test_images_dir="../input/aptos2019-blindness-detection/test_images"

train_df=pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

test_df=pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
train_df.diagnosis.value_counts()
train_df.hist()
import tensorflow 

from tensorflow.python.keras.applications import ResNet50, InceptionV3, Xception

print(os.listdir(("../input/keras-pretrained-models/")))
import tensorflow as tf

IMG_SIZE=256

model_inception_v3 = InceptionV3(

    weights="../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", 

    include_top=False, 

    input_tensor=tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

)
model_inception_v3.trainable=False

model_inception_v3.summary()
model=tf.keras.models.Sequential()

model.add(model_inception_v3)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256,activation='relu',input_dim=6*6*2048))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Dense(5,activation='softmax'))
batch_size=32

seed=666



train_df.id_code=train_df.id_code.apply(lambda x :x+'.png')

test_df.id_code=test_df.id_code.apply(lambda x :x+'.png')

train_df['diagnosis']=train_df['diagnosis'].astype(str)

x_train,x_val=train_test_split(train_df,test_size=0.2,random_state=seed)

train_datagen=image.ImageDataGenerator(rescale=1./255,

                                       horizontal_flip=True,rotation_range=40,

                                       width_shift_range=0.2,shear_range=0.2,

                                      zoom_range=0.2,fill_mode='nearest')



val_datagen=image.ImageDataGenerator(rescale=1./255)

    

test_datagen=image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(

    dataframe=x_train, 

    directory=train_images_dir,

    x_col='id_code',

    y_col='diagnosis',

    target_size=(IMG_SIZE,IMG_SIZE),

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    seed=seed)

validation_generator = val_datagen.flow_from_dataframe(

    dataframe=x_val, 

    directory=train_images_dir,

    x_col='id_code',

    y_col='diagnosis',

    target_size=(IMG_SIZE,IMG_SIZE),

    color_mode='rgb',

    class_mode='categorical',

    batch_size=batch_size,

    shuffle=False,

    seed=seed)

test_generator=test_datagen.flow_from_dataframe(

    dataframe=test_df,

    directory=test_images_dir,

    x_col='id_code',

    y_col=None,

    target_size=(IMG_SIZE,IMG_SIZE),

    color_mode='rgb',

    class_mode=None,

    batch_size=batch_size,

    shuffle=False,

    seed=seed)
callback_list=[tf.keras.callbacks.ReduceLROnPlateau

               (monitor='val_loss',

                factor=0.1,

                patience=5)]
# from keras.engine.topology import Layer

# class MixUpSoftmaxLoss(Layer):

#     def __init__(self, crit, reduction='mean'):

#         super().__init__()

#         self.crit = crit

#         setattr(self.crit, 'reduction', 'none')

#         self.reduction = reduction



#     def forward(self, output, target):

#         if len(target.size()) == 2:

#             loss1 = self.crit(output, target[:, 0].long())

#             loss2 = self.crit(output, target[:, 1].long())

#             lambda_ = target[:, 2]

#             d = (loss1 * lambda_ + loss2 * (1-lambda_)).mean()

#         else:

#             # This handles the cases without MixUp for backward compatibility

#             d = self.crit(output, target)

#         if self.reduction == 'mean':

#             return d.mean()

#         elif self.reduction == 'sum':

#             return d.sum()

#         return d
model.compile(optimizer=tf.train.RMSPropOptimizer(2e-4),

             loss='categorical_crossentropy',

             metrics=['acc'])

history=model.fit_generator(train_generator,  

                steps_per_epoch=math.ceil(len(x_train)/batch_size),

                 epochs=20,

                 callbacks=callback_list,

                 validation_data=validation_generator,

                 validation_steps=32

                         )
# acc=history.history['acc']

# val_acc=history.history['val_acc']

# loss=history.history['loss']

# val_loss=history.history['val_loss']



# epochs=range(1,len(acc)+1)



# plt.plot(epochs,acc,'bo',label='training acc')

# plt.plot(epochs,val_acc,'r',label='val acc')

# plt.title('training and validation accuracy')

# plt.legend()



# plt.figure()



# plt.plot(epochs,loss,'bo',label='training loss')

# plt.plot(epochs,val_loss,'r',label='val loss')

# plt.title('training and validation loss')

# plt.legend()

# plt.show()
from tqdm import tqdm

tta_steps = 10

step=math.ceil(len(test_df)/batch_size)

preds_tta=[]

for i in tqdm(range(tta_steps)):

    test_generator.reset()

    preds = model.predict_generator(generator=test_generator,steps =step)

    preds_tta.append(preds)
pred_mean = np.mean(preds_tta, axis=0)

pred_argmax = np.argmax(pred_mean, axis=1)

test_df['diagnosis']=pred_argmax.astype(int)

test_df.to_csv('submission.csv',index=False)
test_df.diagnosis.value_counts()