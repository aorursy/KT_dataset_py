import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from glob import glob



import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt



from tensorflow.keras import layers
# os.listdir('train')
from zipfile import ZipFile

# with ZipFile('../input//aerial-cactus-identification.zip')as zip_obj :

#   zip_obj.extractall()

with ZipFile('../input/aerial-cactus-identification/test.zip')as test_obj :

  test_obj.extractall()

with ZipFile('../input/aerial-cactus-identification/train.zip')as train_obj :

  train_obj.extractall()
df = pd.read_csv('../input/aerial-cactus-identification/train.csv')

print(df.head())



file_list = df['id']

has_cactus = df['has_cactus']

print(len(file_list), len(has_cactus))
test_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

print(test_df.head())



test_fnames = test_df['id']

test_labels = test_df['has_cactus']



print(len(test_fnames), len(test_labels))
data_paths = glob('train/*.jpg')

test_paths = glob('test/*.jpg')

print(len(data_paths), len(test_paths))
pa = glob('train/*.jpg')[0]

pa

g = tf.io.read_file(pa)

im = tf.io.decode_image(g)

print(im.shape)

plt.imshow(im)

plt.show()
input_shape = (32, 32, 3)

batch_size = 32
data_dir = 'train'



data_paths = []

for fname, label in zip(file_list, has_cactus) :

  data_paths.append((os.path.join(data_dir, fname), label))
data_paths[:10]
def tmp_func (path_name) :

  return path_name
def read_data(path_name) :

  img_path = path_name[0]

  label = tf.strings.to_number(path_name[1], out_type=tf.int64)



  gfile = tf.io.read_file(img_path)

  image = tf.io.decode_image(gfile)

  

  return image, label
a = tf.data.Dataset.from_tensor_slices(np.array(data_paths[:2]))

a = a.map(tmp_func)

p = next(iter(a))

p[0], p[1]
train_ratio = 0.8



train_paths = data_paths[:int(train_ratio*len(data_paths))]

test_paths = data_paths[int(train_ratio*len(data_paths)):]
train_ds = tf.data.Dataset.from_tensor_slices(np.array(train_paths))

train_ds = train_ds.map(read_data)

# print(next(iter(train_ds)))

# 이때 확인하면 image에 대한 tensor값과 그 이미지에 대한 라벨을 알 수 있음

train_ds = train_ds.shuffle(len(train_paths))

# shuffle에 대한 buffer_size에 관해서도(위에 참고)

train_ds = train_ds.batch(batch_size)

train_ds = train_ds.repeat()
valid_ds = tf.data.Dataset.from_tensor_slices(np.array(test_paths))

valid_ds = valid_ds.map(read_data)

valid_ds = valid_ds.batch(batch_size)

valid_ds = valid_ds.repeat()
inputs = layers.Input(input_shape)



# Feature extraction

net = layers.Conv2D(32, 3, 1, 'SAME')(inputs)

net = layers.Activation('relu')(net)

net = layers.Conv2D(32, 3, 1, 'SAME')(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D((2, 2))(net)

net = layers.Dropout(0.5)(net)



net = layers.Conv2D(64, 3, 1, 'SAME')(net)

net = layers.Activation('relu')(net)

net = layers.Conv2D(64, 3, 1, 'SAME')(net)

net = layers.Activation('relu')(net)

net = layers.MaxPooling2D((2, 2))(net)

net = layers.Dropout(0.5)(net)



# classification

net = layers.Flatten()(net)

net = layers.Dense(512)(net)

net = layers.Activation('relu')(net)

net = layers.Dropout(0.5)(net)

net = layers.Dense(1)(net)

net = layers.Activation('sigmoid')(net)



model = tf.keras.Model(inputs=inputs, outputs=net, name='cactus_cnn')
model.summary()
model.compile(loss = tf.keras.losses.binary_crossentropy,

              optimizer = tf.keras.optimizers.Adam(),

              metrics=['accuracy'])
steps_per_epoch = len(train_paths) // batch_size

validation_steps = len(test_paths) // batch_size
hist = model.fit(train_ds,

                 validation_data=valid_ds,

                 validation_steps=validation_steps,

                 steps_per_epoch=steps_per_epoch,

                 epochs = 30)
test_df.head()
test_fnames[:5]
test_labels[:5]
test_dir = 'test'



eval_paths = []

for fname in test_fnames :

  eval_paths.append(os.path.join(test_dir, fname))



print(eval_paths[:5])
def image_read(path) :

  g = tf.io.read_file(path)

  im = tf.io.decode_image(g)



  return im
from tqdm import tqdm_notebook
# test_images = [image_read(path) for path in eval_paths]



test_images = []

for path in tqdm_notebook(eval_paths) :

  test_images.append(image_read(path))



# np.array(test_image).shape
test_ds = tf.data.Dataset.from_tensor_slices(eval_paths)

test_ds = test_ds.map(image_read)

test_ds = test_ds.batch(batch_size)
pred = model.predict(test_ds)
pred.shape
pred = pred.reshape((4000))
submit_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

test_fnames = submit_df['id']

test_labels = pred  # 결과가 onehot이 아닌 binary로 담아줘야함



submit_file = pd.DataFrame({'id':test_fnames, 'has_cactus':test_labels}, columns = ['id', 'has_cactus'])

submit_file

submit_file.to_csv('submission.csv', index=False)