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
!pip install git+https://github.com/qubvel/efficientnet

from efficientnet.keras import EfficientNetB7

from efficientnet.keras import EfficientNetB3

import zipfile
import pandas as pd
import keras
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
train_path = '../input/shopeepd-filtered/data_light/train/'

test_path = '../input/shopeepd-filtered/data_light/test/'

df = pd.read_csv("../input/shopeepd-filtered/data_light/train.csv")

df = df.sample(frac=1)

df.head()

test_df = pd.read_csv("../input/shopeepd-filtered/data_light/test.csv")
test_df.head()

df['category'] = df['category'].astype(int)
df['category'] = df['category'].apply(lambda x: "{:02d}".format(x)).astype(str)
cats = df['category']
df['combined_filename'] = df['category'].map(lambda x: x + "/").astype(str) + df['filename']
df.head()
mod_df = df.drop(columns = ["filename"])
mod_df.head()
mod_df = mod_df.sample(frac=1)
mod_df.head()
for cat in cats:
  path = os.path.join(train_path, cat)
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    plt.imshow(img_array, cmap='gray')
    plt.show()
    break
  break
new_array = cv2.resize(img_array, (256,256))
plt.imshow(new_array)
plt.show()
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.15)
IMG_SIZE = 300
train_generator=datagen.flow_from_dataframe(
dataframe=mod_df,
directory=train_path,
x_col="combined_filename",
y_col="category",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMG_SIZE,IMG_SIZE))
valid_generator=datagen.flow_from_dataframe(
dataframe=mod_df,
directory=train_path,
x_col="combined_filename",
y_col="category",
subset="validation",
batch_size=32,
seed=40,
shuffle=True,
class_mode="categorical",
target_size=(IMG_SIZE,IMG_SIZE))
test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=test_df,
directory=test_path,
x_col="filename",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(IMG_SIZE,IMG_SIZE))
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = EfficientNetB7(input_shape=IMG_SHAPE,
                           include_top=False,
                            weights='imagenet')
base_model.trainable = False

import tensorflow as tf
import zipfile
import pandas as pd
import keras
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(42)
from keras.optimizers import Adam
new_model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
new_model.compile(Adam(lr=3e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size + 1
new_model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=4
)
from keras.models import load_model

new_model=load_model("../input/output/256x256_efficientnet_9epochs_0.15valid")
ls
base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 600
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True
new_model.compile(Adam(lr=5e-5),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
new_model.summary()
mod_df = mod_df.sample(frac=1)
train_generator=datagen.flow_from_dataframe(
dataframe=mod_df,
directory=train_path,
x_col="combined_filename",
y_col="category",
subset="training",
batch_size=32,
seed=42,
shuffle=True,
class_mode="categorical",
target_size=(IMG_SIZE,IMG_SIZE))
valid_generator=datagen.flow_from_dataframe(
dataframe=mod_df,
directory=train_path,
x_col="combined_filename",
y_col="category",
subset="validation",
batch_size=32,
seed=40,
shuffle=True,
class_mode="categorical",
target_size=(IMG_SIZE,IMG_SIZE))
from tensorflow import keras

new_model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=6
)

test_generator.reset()
pred=new_model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
import numpy as np
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
res = test_df.drop(columns='category')
res['category'] = pd.DataFrame(predictions).astype(int)
res.head()
res.to_csv("submit.csv", index=False)
ls
final_model = new_model
final_model.save("256x256EfficientNetB7")
final_model.save_weights("256x256_8epochs_0.15valid_weights")
ls
corr_df = df
corr_df = corr_df.sample(frac=1)
corr_df
corr_datagen=ImageDataGenerator(rescale=1./255.)
corr_generator=corr_datagen.flow_from_dataframe(
dataframe=corr_df,
directory=train_path,
x_col="combined_filename",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(IMG_SIZE,IMG_SIZE))
STEP_SIZE_CORR = corr_generator.n//corr_generator.batch_size + 1
corr_generator.reset()
corr_pred=new_model.predict_generator(corr_generator,
steps=STEP_SIZE_CORR,
verbose=1)
corr_predicted_class_indices=np.argmax(corr_pred,axis=1)
corr_labels = (train_generator.class_indices)
corr_labels = dict((v,k) for k,v in corr_labels.items())
corr_predictions = [labels[k] for k in corr_predicted_class_indices]
print(len(corr_predictions))
df['category'].value_counts()
corr_df['category'] = pd.DataFrame(corr_predictions).astype(int)
corr_df['category'].value_counts()
corr_df.drop(columns=['category'])
corr_df['category'] = pd.DataFrame(corr_predictions).astype(int)
corr_df
corr_df.to_csv("submit1.csv", index=False)
ls
