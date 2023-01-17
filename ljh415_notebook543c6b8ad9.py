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
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model

from glob import glob
%matplotlib inline
from zipfile import ZipFile
# with ZipFile('../input//aerial-cactus-identification.zip')as zip_obj :
#   zip_obj.extractall()
with ZipFile('../input/aerial-cactus-identification/test.zip')as test_obj :
  test_obj.extractall()
with ZipFile('../input/aerial-cactus-identification/train.zip')as train_obj :
  train_obj.extractall()
train_df = pd.read_csv('../input/aerial-cactus-identification/train.csv')
train_df.head()
train_df["has_cactus"].value_counts()
os.listdir()
img_gen = ImageDataGenerator(rescale=1/255.)
train_dir = 'train/'
batch_size = 64
image_size = 32
train_df.has_cactus = train_df.has_cactus.astype(str)

train_generator = img_gen.flow_from_dataframe(dataframe=train_df[:14001], directory=train_dir, x_col='id',
                                             y_col='has_cactus', class_mode='binary', batch_size=batch_size, 
                                             target_size=(image_size, image_size))

validation_generator = img_gen.flow_from_dataframe(dataframe=train_df[14001:], directory=train_dir, x_col='id',
                                                  y_col='has_cactus', class_mode='binary', batch_size=batch_size,
                                                  target_size=(image_size, image_size))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in base_model.layers:
    layer.trainable = True
    
net = base_model.output

net = Flatten()(net)
net = Dense(512, activation='relu')(net)
net = Dense(1, activation='sigmoid', name='ResNet')(net)

model = Model(inputs=base_model.input, outputs=net)

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
             metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
steps_per_epoch = train_generator.n//batch_size
validation_steps = validation_generator.n//batch_size
hist = model.fit(train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                epochs = 30,
                callbacks=[es])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title("Accuracy for every epoch")
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['train','validation'],loc='lower right')
plt.show()
import os
test_dir = "test"
test_df=pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")
os.listdir('test')
from skimage import io
for _ , _, files in os.walk(test_dir):
    i=0
    for file in files:
        image=io.imread(os.path.join(test_dir, file))
        test_df.iloc[i,0]=file
        image=image.astype(np.float32)/255.0
        test_df.iloc[i,1]=model.predict(image.reshape((1, 32, 32, 3)))[0][0]
        i+=1
! rm -r *
test_df.to_csv("submission.csv",index=False)
test_df.head()
