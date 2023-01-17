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

from glob import glob
%matplotlib inline
from zipfile import ZipFile
# with ZipFile('../input//aerial-cactus-identification.zip')as zip_obj :
#   zip_obj.extractall()
with ZipFile('../input/aerial-cactus-identification/test.zip')as test_obj :
  test_obj.extractall()
with ZipFile('../input/aerial-cactus-identification/train.zip')as train_obj :
  train_obj.extractall()
os.listdir('../input/aerial-cactus-identification/')
train_csv = pd.read_csv('../input/aerial-cactus-identification/train.csv')
train_csv.head()
sub = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')
sub.head()
train_img_id = train_csv['id']
train_img_label = train_csv['has_cactus']
len(train_img_id), len(train_img_label)
input_paths = []
for fname, label in tqdm(zip(train_img_id, train_img_label)):
    input_paths.append((os.path.join('train', fname), label))
    
len(input_paths)
train, valid = train_test_split(input_paths, train_size=0.8)
len(train), len(valid)
def read_img(data):
    img_path = data[0]
    label = data[1]
    label = tf.strings.to_number(label, out_type=tf.int64)
    
    tf_img = tf.io.read_file(img_path)
    img = tf.image.decode_image(tf_img)
    
    return img, label
train_dataset = tf.data.Dataset.from_tensor_slices(np.array(train))
train_dataset = train_dataset.map(read_img)
train_dataset = train_dataset.shuffle(len(train))
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.repeat()
valid_dataset = tf.data.Dataset.from_tensor_slices(np.array(valid))
valid_dataset = valid_dataset.map(read_img)
valid_dataset = valid_dataset.batch(32)
valid_dataset = valid_dataset.repeat()
inputs = Input((32, 32, 3))

# Feature Extraction
net = Conv2D(32, 3, 1, 'SAME')(inputs)
net = Activation('relu')(net)
net = Conv2D(32, 3, 1, 'SAME')(net)
net = Activation('relu')(net)
net = MaxPooling2D((2,2))(net)
net = BatchNormalization()(net)

net = Conv2D(64, 3, 1, 'SAME')(net)
net = Activation('relu')(net)
net = Conv2D(64, 3, 1, 'SAME')(net)
net = Activation('relu')(net)
net = MaxPooling2D((2,2))(net)
net = BatchNormalization()(net)

# classification
net = Flatten()(net)
net = Dense(512)(net)
net = Activation('relu')(net)
net = BatchNormalization()(net)
net = Dense(1)(net)
output = Activation('sigmoid')(net)

basic_cnn = tf.keras.Model(inputs=inputs, outputs = output, name='basic_cnn')

basic_cnn.summary()
basic_cnn.compile(loss = tf.keras.losses.binary_crossentropy,
             optimizer = tf.keras.optimizers.Adam(),
             metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
# mc = ModelCheckpoint('basic_cnn.h5', monitor='val_accuracy', save_best_only=True)
steps_per_epoch = len(train) // 32
validation_steps = len(valid) // 32

hist = basic_cnn.fit(train_dataset,
                validation_data = valid_dataset,
                validation_steps=validation_steps,
                steps_per_epoch=steps_per_epoch,
                epochs=50,
                callbacks=[es]
                )

test_imgs = glob('test/*')
len(test_imgs)
def test_img_read(path) :
    tf_img = tf.io.read_file(path)
    img = tf.io.decode_image(tf_img)
    return img
test_ds = tf.data.Dataset.from_tensor_slices(test_imgs)
test_ds = test_ds.map(test_img_read)
test_ds = test_ds.batch(32)
pred = basic_cnn.predict(test_ds)
pred = pred.reshape((4000))
os.listdir('../working')
os.mkdir('output')
!rm -r *
test_fname = sub['id']
test_label = pred

sub_file = pd.DataFrame({'id':test_fname, 'has_cactus':test_label}, columns=['id', 'has_cactus'])
sub_file.to_csv('./submission.csv', index=False)
