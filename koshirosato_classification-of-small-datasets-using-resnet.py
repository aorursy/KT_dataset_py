import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import *

from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

from tqdm import tqdm
SEED = 42

EPOCHS = 50

BATCH_SIZE = 32 

IMG_SIZE = 256

ROOT = '../input/flower-color-images/flower_images/flower_images/'



df = pd.read_csv(ROOT + 'flower_labels.csv')
def seed_everything(seed):

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



seed_everything(SEED)
df = df.replace({0:'phlox',1:'rose',2:'calendula',3:'iris',4:'leucanthemum maximum',

                 5:'bellflower',6:'viola',7:'rudbeckia laciniata',

                 8:'peony',9:'aquilegia'})
df.head()
df.label.value_counts().plot.bar()
def img_plot(df):

    imgs = []

    labels = []

    df = df.sample(frac=1)

    for file, label in zip(df['file'][:25], df['label'][:25]):

        img = cv2.imread(ROOT+file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgs.append(img)

        labels.append(label)

    f, ax = plt.subplots(5, 5, figsize=(15,15))

    for i, img in enumerate(imgs):

        ax[i//5, i%5].imshow(img)

        ax[i//5, i%5].axis('off')

        ax[i//5, i%5].set_title(labels[i])

    plt.show()



img_plot(df)
train_df, test_df = train_test_split(df, 

                                     test_size=0.5, 

                                     random_state=SEED, 

                                     stratify=df['label'].values)







def create_datasets(df, img_size):

    imgs = []

    for file in tqdm(df['file']):

        img = cv2.imread(ROOT+file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (img_size,img_size))

        imgs.append(img)

    # not normalize    

    imgs = np.array(imgs)

    df = pd.get_dummies(df['label'])

    return imgs, df





train_imgs, train_df = create_datasets(train_df, IMG_SIZE)

test_imgs, test_df = create_datasets(test_df, IMG_SIZE)
num_classes = len(df.label.value_counts())



def build_model(ResNet, img_size, n):

    inp = Input(shape=(img_size,img_size, n))

    resnet = ResNet(input_shape=(img_size,img_size,n),

                    weights='imagenet',

                    include_top=False)

    # freeze ResNet

    resnet.trainable = False

    x = resnet(inp)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    x = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inp, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model





resnet50 = build_model(ResNet50, IMG_SIZE, 3)

resnet50.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint('resnet50.h5', 

                                                monitor='loss', 

                                                save_best_only=True,

                                                save_weights_only=True)



resnet50.fit(train_imgs, train_df, batch_size=BATCH_SIZE,

          epochs=EPOCHS, verbose=0, callbacks=[checkpoint])

resnet50.load_weights('resnet50.h5')





resnet50.evaluate(test_imgs, test_df)
resnet101 = build_model(ResNet101, IMG_SIZE, 3)

resnet101.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint('resnet101.h5', 

                                                monitor='loss', 

                                                save_best_only=True,

                                                save_weights_only=True)



resnet101.fit(train_imgs, train_df, batch_size=BATCH_SIZE,

              epochs=EPOCHS, verbose=0, callbacks=[checkpoint])

resnet101.load_weights('resnet101.h5')



resnet101.evaluate(test_imgs, test_df)
resnet152 = build_model(ResNet152, IMG_SIZE, 3)

resnet152.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint('resnet152.h5', 

                                                monitor='loss', 

                                                save_best_only=True,

                                                save_weights_only=True)



resnet152.fit(train_imgs, train_df, batch_size=BATCH_SIZE,

              epochs=EPOCHS, verbose=0, callbacks=[checkpoint])

resnet152.load_weights('resnet152.h5')



resnet152.evaluate(test_imgs, test_df)