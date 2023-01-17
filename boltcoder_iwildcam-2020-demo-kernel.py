# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import json
import os
from IPython.display import FileLink
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as f:
    train_data = json.load(f)
    
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as f:
    test_data = json.load(f)
train_data.keys()
train = pd.DataFrame(train_data['annotations'])
train.head()
train.rename(columns={'count': 'cnt'}, inplace=True)

train[train.cnt > 1].describe()
train.describe()
train_img = pd.DataFrame(train_data['images'])
indices1 = []
indices2 = []
indices1.append( train[ train['image_id'] == '896c1198-21bc-11ea-a13a-137349068a90' ].index )
indices1.append( train[ train['image_id'] == '8792549a-21bc-11ea-a13a-137349068a90' ].index )
indices1.append( train[ train['image_id'] == '87022118-21bc-11ea-a13a-137349068a90' ].index )
indices1.append( train[ train['image_id'] == '98a295ba-21bc-11ea-a13a-137349068a90' ].index )
indices2.append( train_img[ train_img['id'] == '896c1198-21bc-11ea-a13a-137349068a90' ].index )
indices2.append( train_img[ train_img['id'] == '8792549a-21bc-11ea-a13a-137349068a90' ].index )
indices2.append( train_img[ train_img['id'] == '87022118-21bc-11ea-a13a-137349068a90' ].index )
indices2.append( train_img[ train_img['id'] == '98a295ba-21bc-11ea-a13a-137349068a90' ].index )

for _id in train_img[train_img['location'] == 537]['id'].values:
    indices1.append( train[ train['image_id'] == _id ].index )
    indices2.append(train_img[ train_img['id'] == _id ].index)
for the_index in indices1:
    train = train.drop(train.index[the_index])
for the_index in indices2:
    train_img = train_img.drop(train_img.index[the_index])

train_img.head()
fig = plt.figure(figsize=(19, 4))
ax = sns.distplot(train['category_id'])
plt.title('distribution of number of data per category')
fig = plt.figure(figsize=(30, 4))
ax = sns.barplot(x="category_id", y="cnt",data=train)
plt.title('distribution of count per id')
fig = plt.figure(figsize=(30, 4))
ax = sns.countplot(train_img['location'])
plt.title('distribution of number of animals by location')
labels_month = sorted(list(set(train_img['datetime'].map(lambda str: str[5:7]))))
# fig, ax = plt.subplots(1,2, figsize=(20,7)
plt.title('Count of train data per month')
ax = sns.countplot(train_img['datetime'].map(lambda str: str[5:7] ), order=labels_month)
ax.set(xlabel='Month', ylabel='count')
# ax.set(ylim=(0,55000))
train_img.describe()
train.describe()
train_img = train_img
train = train
train_img['category'] = train['category_id']
train_img.drop(train_img.columns.difference(['file_name','category']), 1, inplace=True)
train_img['category'] = train_img['category'].apply(str)
train_img.head()
train_img[ train_img['file_name'] == '883572ba-21bc-11ea-a13a-137349068a90.jpg' ].index
train_img.drop(123658,inplace=True)
train_img.drop(123651,inplace=True)
train_img.drop(123653,inplace=True)
# !pip install tensorflow-gpu==1.14.0
# !pip install keras==2.2.4

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
# import pickle
import dill
from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip = True,    
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
   rotation_range = 40,
   shear_range = 0.3,
   channel_shift_range=150.0,
   fill_mode='nearest',
   brightness_range=(0.2, 0.9)
)
# (max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
#                       p_affine=1., p_lighting=1.
train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_img[90000:120000],
        directory='/kaggle/input/iwildcam-2020-fgvc7/train',
        x_col="file_name",
        y_col="category",
        target_size=(150,150),
        batch_size=256,
        classes = train_img['category'].unique().tolist(),
        class_mode='categorical')
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)
# cache_dir = expanduser(join('~', '.keras'))
# if not exists(cache_dir):
#     makedirs(cache_dir)
# models_dir = join(cache_dir, 'models')
# if not exists(models_dir):
#     makedirs(models_dir)
    
# !cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
# !cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
# !cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
!ls ../input/keras-pretrained-models/ 
# !git clone https://github.com/qubvel/efficientnet.git
# import efficientnet.efficientnet.tfkeras as efn
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
pre_trained_model = tf.keras.applications.InceptionV3(include_top=False,input_shape = (150, 150, 3),
                                                weights='../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
# pre_trained_model = efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(96, 96, 3))
for layer in pre_trained_model.layers:
    layer.trainable = False
# x = pre_trained_model.output
# predictions = Dense(573, activation="softmax")(x)
# model = Model(inputs=pre_trained_model.input, outputs=predictions)
model = Sequential()
    # first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(216,activation='softmax'))
pretrainedInput = pre_trained_model.input
pretrainedOutput = pre_trained_model.output
output = model(pretrainedOutput)
model = Model(pretrainedInput, output)
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = new_model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size+1,
        epochs=5,
        shuffle = True,
        verbose = 1)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy vs epochs')
plt.legend(loc=0)
plt.figure()


plt.show()
new_model.save('Modeln.h5')
FileLink('Modeln.h5')
test = pd.DataFrame(test_data['images'])
test.head()
test.describe()
test_data.keys()
test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory='/kaggle/input/iwildcam-2020-fgvc7/test',
        x_col="file_name",
        target_size=(150, 150),
        batch_size=64,class_mode=None)
new_model = tf.keras.models.load_model('/kaggle/input/model-1/Modeln.h5')
preds = new_model.predict_generator(test_generator,
steps=test_generator.n//test_generator.batch_size+1,
verbose=1)

predicted_class_indices=np.argmax(preds,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
Id=test.id
results=pd.DataFrame({"Id":Id,
                      "Category":predictions})
submission = pd.read_csv('/kaggle/input/iwildcam-2020-fgvc7/sample_submission.csv')
submission = submission.drop(['Category'], axis=1)
submission = submission.merge(results, on='Id')
submission.to_csv('modeln.csv', index=False)
FileLink('modeln.csv')

# results.to_csv("results.csv",index=False)