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
import numpy as np
import pandas  as pd
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")
train
train['filename'] = train.id.str[0] + "/" + train.id.str[1] + "/" + train.id.str[2] + "/" + train.id + ".jpg"
train
train["label"] = train['landmark_id'].astype(str)
train
sub = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")
sub['filename'] = sub.id.str[0] + "/" + sub.id.str[1] + "/" + sub.id.str[2] + "/" + sub.id + "jpg"
sub
y = train['landmark_id'].values
y
num_classes = np.max(y)
num_classes
from collections import Counter
count = Counter(y).most_common(1000)
count
k_labels = [c[0] for c in count]
train_keep = train[train['landmark_id'].isin(k_labels)]
train_keep
val_rate = 0.25
batch_size = 32
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(validation_split=val_rate)
dir = "../input/landmark-recognition-2020/train/"
train_gen = datagen.flow_from_dataframe(train_keep, directory=dir, x_col="filename", y_col="label", weight_col=None, 
                                        target_size=(256, 256), color_mode="rgb", classes=None, class_mode="categorical",
                                       batch_size=batch_size, shuffle=True, subset="training", interpolation="nearest",
                                       validate_filenames=False)
val_gen = datagen.flow_from_dataframe(train_keep, directory=dir, x_col="filename", y_col="label", weight_col=None,
                                     target_size=(256, 256), color_mode="rgb",classes=None, class_mode="categorical", 
                                     batch_size=batch_size, shuffle=True, subset="validation",interpolation="nearest", 
                                     validate_filenames=False)
from keras.applications import MobileNetV2
from keras.utils import to_categorical
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_sparse_categorical_accuracy', patience = 3, verbose = 1, 
                                           factor = 0.2, min_lr = 0.00001)

optimizer = Adam(lr = .0001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = .0, amsgrad = False)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.utils import to_categorical
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.applications.xception import Xception
import tensorflow as tf
import tensorflow.keras.layers as L
# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
#     print('Running on TPU ', tpu.master())
# except ValueError:
#     tpu = None

# if tpu:
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
# else:
#     strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

# print("REPLICAS: ", strategy.num_replicas_in_sync)
# with strategy.scope():
#     pretrained_model = tf.keras.applications.ResNet50V2(
#     weights='imagenet',
#     include_top=False ,
#     input_shape=(256, 256,3)
#     )
#     pretrained_model.trainable = False
    
#     model = tf.keras.Sequential([
#         pretrained_model,

#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(1000, activation='softmax')
#     ])
#     model.compile(
#         optimizer=optimizer,
#         loss = 'categorical_crossentropy',
#         metrics=['categorical_accuracy']
#     )
!pip install -q efficientnet
import efficientnet.tfkeras as efn
model = tf.keras.Sequential([
    efn.EfficientNetB3(
        input_shape=(256, 256, 3),
        weights='imagenet',
        include_top=False
    ),
    L.GlobalAveragePooling2D(),
    L.Dense(1000, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['categorical_accuracy']
)
# training parameters
train_steps = int(len(train_keep)*(1-val_rate))//batch_size
val_steps = int(len(train_keep)*val_rate)//batch_size

model_checkpoint = ModelCheckpoint("model_efnB3.h5", save_best_only=True, verbose=1)
history = model.fit_generator(train_gen, 
                              steps_per_epoch=train_steps, 
                              epochs=1,validation_data=val_gen, 
                              validation_steps=val_steps,
                              callbacks=[model_checkpoint])
sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")
sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"
sub

test_gen = ImageDataGenerator().flow_from_dataframe(
    sub,
    directory="/kaggle/input/landmark-recognition-2020/test/",
    x_col="filename",
    y_col=None,
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode=None,
    batch_size=1,
    shuffle=True,
    subset=None,
    interpolation="nearest",
    validate_filenames=False)
y_pred_one_hot = model.predict_generator(test_gen, verbose=1, steps=len(sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)
y_prob = np.max(y_pred_one_hot, axis=-1)
print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(train_keep.landmark_id.values)

y_pred = [y_uniq[Y] for Y in y_pred]
for i in range(len(sub)):
    sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])
sub = sub.drop(columns="filename")
sub.to_csv("submission.csv", index=False)
sub

