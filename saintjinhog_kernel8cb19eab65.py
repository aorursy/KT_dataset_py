import os 

from glob import glob

import shutil 

from tqdm import tqdm



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image



import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_dir = '../input/train/train/'

test_dir = '../input/test/test'







input_shape = (32, 32, 3)

batch_size = 32

num_classes = 2

num_epochs = 1

data_augmentation = True



learning_rate = 0.001



def build_model(input_shape):

    inputs = layers.Input(input_shape)

    net = layers.Conv2D(32, (3, 3), padding='same')(inputs)

    net = layers.Activation('relu')(net)

    net = layers.Conv2D(32, (3, 3))(net)

    net = layers.Activation('relu')(net)

    net = layers.MaxPooling2D(pool_size=(2, 2))(net)

    net = layers.Dropout(0.25)(net)

    

    net = layers.Conv2D(64, (3, 3), padding='same')(net)

    net = layers.Activation('relu')(net)

    net = layers.Conv2D(64, (3, 3))(net)

    net = layers.Activation('relu')(net)

    net = layers.MaxPooling2D(pool_size=(2, 2))(net)

    net = layers.Dropout(0.25)(net)



    net = layers.Flatten()(net)

    net = layers.Dense(512)(net)

    net = layers.Activation('relu')(net)

    net = layers.Dropout(0.5)(net)

    net = layers.Dense(num_classes)(net)

    net = layers.Activation('softmax')(net)



    return tf.keras.Model(inputs=inputs, outputs=net)





model = build_model(input_shape)

model.compile(loss='categorical_crossentropy',

              optimizer=tf.keras.optimizers.Adam(learning_rate),

              metrics=['accuracy'])





df = pd.read_csv('../input/train.csv')

df.head()



#  (fname, cls), (fname, cls), (fname, cls), (fname, cls) (fname, cls)

datasets = []

for fname, cls in zip(df['id'], df['has_cactus']):

    datasets.append((fname, cls))



np.random.shuffle(datasets)

train_paths = datasets[:int(len(datasets) * 0.8)]

test_paths = datasets[int(len(datasets) * 0.8):]



# train_rate = 0.8

# np.random.shuffle(datasets)

# num_trains = int(len(datasets) * train_rate)



# train_paths = datasets[:num_trains]

# test_paths = datasets[num_trains:]



def batch_dataset(batch_paths):

    batch_images = []

    batch_labels = []



    for fname, cls in batch_paths:

        img_path = os.path.join(train_dir, fname)

        image = np.array(Image.open(img_path))

        label = np.array(np.array([0, 1]) == cls).astype(np.uint8)

        batch_images.append(image)

        batch_labels.append(label)



    batch_images = np.array(batch_images)

    batch_labels = np.array(batch_labels)

    return batch_images, batch_labels



images, labels = batch_dataset(train_paths[:4])

images.shape, labels.shape



for epoch in range(num_epochs):

    np.random.shuffle(train_paths)

    batches_per_epoch = len(train_paths) // batch_size

    for step in tqdm(range(batches_per_epoch)):

        images, labels = batch_dataset(train_paths[step*batch_size: (step+batch_size)*batch_size])

        model.fit(images, labels, batch_size=batch_size, verbose=0)

    

#     # Evaluate

#     batches_per_epoch = len(test_paths) // batch_size

#     test_imgs, test_lbls = batch_dataset(test_paths[step*batch_size: (step+batch_size)*batch_size])

#     print(model.evaluate(test_imgs, test_lbls, verbose=0))



test_df = pd.read_csv('../input/sample_submission.csv')

test_df['id'][0]



answers = []

for fname in test_df['id']:

    path = os.path.join(test_dir, fname)

    image = np.array(Image.open(path))

    image = np.expand_dims(image, 0)

    logit = model.predict(image)

    ans = logit[0, 1]

    answers.append(ans)



submit_data = {'id': test_df['id'],

               'has_cactus': answers}



submit_df = pd.DataFrame(submit_data)

submit_df.to_csv('samplesubmission.csv', index=False)