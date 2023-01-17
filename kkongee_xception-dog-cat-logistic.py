# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import gc

import os

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm



from keras import backend as K

warnings.filterwarnings(action='ignore')



K.image_data_format()
DATA_PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'



import re

list_for_df = []

for traindata in os.listdir(os.path.join(DATA_PATH, 'train/train')):

#   print(traindata)

  tmplist = []

  if traindata[0] == 'c':

    tmplist.append(0)

  else:

    tmplist.append(1)

  tmplist.append(traindata)

  list_for_df.append(tmplist)



train_df = pd.DataFrame(data=list_for_df, columns=['class','fname'])

train_df['class'] = train_df['class'].astype('str')

train_df.head()

train_df.shape
train_df.head()
list_for_df = []

for testdata in os.listdir(os.path.join(DATA_PATH, 'test/test')):

#   print(traindata)

  tmplist = []

  tmplist.append(testdata)

  list_for_df.append(tmplist)



df_test = pd.DataFrame(data=list_for_df, columns=['fname'])
df_test.head()
df_test.shape
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(train_df)
df_train.head()
df_val.head()
original_dataset_dir = os.path.join(DATA_PATH, 'train/train')



import PIL

from PIL import ImageDraw



tmp_imgs = df_train['fname'][100:110]

plt.figure(figsize=(12,20))



for num, f_name in enumerate(tmp_imgs):

  img = PIL.Image.open(os.path.join(original_dataset_dir, f_name))

  plt.subplot(5,2,num+1)

  plt.title(f_name)

  plt.imshow(img)

  plt.axis('off')
from keras.applications import Xception



conv_base = Xception(weights='imagenet',

                  include_top=False,

                  input_shape=(224,224,3))



conv_base.summary()
conv_base.trainable = True



set_trainable = False

for layer in conv_base.layers:

  if layer.name.split('_')[0] == 'block1':

    set_trainable = True

  if layer.name.split('_')[0] == 'block2':

    set_trainable = True

  if layer.name.split('_')[0] == 'block13':

    set_trainable = True

  if layer.name.split('_')[0] == 'block14':

    set_trainable = True

  if set_trainable:

    layer.trainable = True

  else:

    layer.trainable = False
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.preprocessing.image import ImageDataGenerator



img_size = (224,224)

nb_train_samples = len(df_train)

nb_validation_samples = len(df_val)

nb_test_samples = len(df_test)

epochs = 12

batch_size = 64



# Define Generator config

train_datagen = ImageDataGenerator(

  horizontal_flip = True,

  vertical_flip = False,

  zoom_range = 0.10,

  rescale=1./255

)



val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



original_dataset_dir = os.path.join(DATA_PATH, 'train/train')



train_generator = train_datagen.flow_from_dataframe(

    dataframe = df_train,

    directory = original_dataset_dir,

    x_col = 'fname',

    y_col = 'class',

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = 'binary',

    batch_size = batch_size,

    seed = 42

)



validation_generator = val_datagen.flow_from_dataframe(

    dataframe = df_val,

    directory = original_dataset_dir,

    x_col = 'fname',

    y_col = 'class',

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = 'binary',

    batch_size = batch_size,

    shuffle = False

)



test_data_path = os.path.join(DATA_PATH, 'test/test')



test_generator = test_datagen.flow_from_dataframe(

    dataframe = df_test,

    directory = test_data_path,

    x_col = 'fname',

    y_col = None,

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = None,

    batch_size = batch_size,

    shuffle = False

)
from sklearn.metrics import f1_score



def micro_f1(y_true, y_pred):

  return f1_score(y_true, y_pred, average='micro')



def get_step(num_samples, batch_size):

  if (num_samples % batch_size) > 0:

    return (num_samples // batch_size) + 1

  else:

    return num_samples // batch_size



import keras



class LossHistory(keras.callbacks.Callback):

  def on_train_begin(self, logs={}):

    self.losses = []

    self.val_losses = []

    self.acces = []

    self.val_acces = []





  def on_batch_end(self, batch, logs={}):

    self.losses.append(logs.get('loss'))

    self.val_losses.append(logs.get('val_loss'))

    self.acces.append(logs.get('acc'))

    self.val_acces.append(logs.get('val_acc'))



loss_history = LossHistory()
def freeze(model):

    """Freeze model weights in every layer."""

    for layer in model.layers:

        layer.trainable = False



        if isinstance(layer, models.Model):

            freeze(layer)
from keras import models

from keras import layers

from keras import optimizers



from keras.callbacks import ModelCheckpoint, EarlyStopping



filepath = "Xception_bin.h5"



es = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')

callbackList = [es, loss_history]





model = models.Sequential()

model.add(conv_base)

model.add(layers.GlobalAveragePooling2D())

# model.add(layers.Flatten())

# model.add(layers.Dropout(0.5))

model.add(layers.Dense(256,activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))

model.add(layers.Activation('sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])



history = model.fit_generator(train_generator,

                   steps_per_epoch=100,

                   epochs=50,

                   validation_data=validation_generator,

                   validation_steps=100,

                   callbacks=callbackList)



# freeze 해줘야함

from keras import models



def freeze(model):

    """Freeze model weights in every layer."""

    for layer in model.layers:

        layer.trainable = False



        if isinstance(layer, models.Model):

            freeze(layer)



freeze(model)

model.save(filepath)
import matplotlib.pyplot as plt



history = model.history



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(loss_history.acces)

plt.plot(loss_history.val_acces)

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(loss_history.losses)

plt.plot(loss_history.val_losses)

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
steps = len(df_test) // batch_size + 1
mypred = model.predict_generator(

    test_generator,

    steps=steps

)
mypred_val = mypred

mypred_val.shape
df_test.head()
id_val = df_test['fname'].str.split(".").str[0].values

id_val
np.shape(id_val)
sptl = df_test['fname'][:30].values
sptl
import matplotlib.pyplot as plt

plt.figure(figsize=(20,40))



testdir = os.path.join(DATA_PATH, 'test/test')



import PIL



num = 0



for fname in sptl:

  img = PIL.Image.open(os.path.join(testdir, fname))

  plt.subplot(8,4,num+1)

  if mypred_val[num]<0.5:

    plt.title('cat')

  else:

    plt.title('dog')

  plt.axis('off')

  plt.imshow(img)

  num += 1
submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
submission['id'] = id_val

submission['label'] = mypred_val
submission.head()
my_submission = submission.set_index('id')
my_submission.head()
my_submission.to_csv("Xception_submit.csv")