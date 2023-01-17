# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os 

from glob import glob



# Input data files are available in the "../input/" directory.



INPUT_PATH = '../input'

IMG_SIZE_LOG2=6

IMG_SIZE=1<<IMG_SIZE_LOG2

print('Mask size will be %dx%d'%(IMG_SIZE,IMG_SIZE))

df_train = pd.read_csv('%s/train_masks.csv'%INPUT_PATH)

df_train = df_train[['img']]

ids_train_tmp = df_train['img'].map(lambda s: s.split('.')[0])

df_train['img'] = ids_train_tmp

df_train['id'] = ids_train_tmp.map(lambda s: s.split('_')[0])

df_train['angle'] = ids_train_tmp.map(lambda s: s.split('_')[1])

df_train['x_angle'] = (df_train['angle'].astype(float) - 1.0) / 16.0

df_train.head()
df_meta = pd.read_csv('%s/metadata.csv'%INPUT_PATH)

df_meta.head()
#normalize data

df_meta['make'] = df_meta['make'].str.lower()

model = df_meta['model'].fillna(df_meta['trim1']).str.lower()

# normalize further

model = model.str.replace('series', '')

model = model.str.replace('plug-in', '')

model = model.str.replace('hybrid', '')

model = model.str.replace(' ev', '')

model = model.str.replace(' el', '')

model = model.str.replace(' limited', '')

model = model.str.replace(' 250c', ' 250')

model = model.str.replace(' 350c', ' 350')

model = model.str.replace('a8 l', 'a8')

model = model.str.replace('cts-v', 'cts')

model = model.str.replace(' esv', '')

model = model.str.replace(' turbo', '')

model = model.str.replace('grand ', '')

model = model.str.replace(' select', '')

model = model.str.replace(' gti', '')

model = model.str.replace(' unlimited', '')

model = model.str.replace('prius c', 'prius')

model = model.str.replace('prius v', 'prius')

model = model.str.replace('slk-class', 'slk')

model = model.str.strip()

car = df_meta['make'].str.lower()+ ' ' + model

car = car.str.replace('dodge ram', 'ram')

car = car.str.replace('ram ram', 'ram')

df_meta['car'] = car



df_meta.head()
df_meta_x=pd.get_dummies(df_meta, prefix='x', columns=['car'])

df_meta_x['x_year']=(2017.0 - df_meta['year']) / 7.0

df_meta_x.set_index(['id'], inplace=True)

df_in = pd.merge(df_train, df_meta_x, left_on='id', right_index=True)

df_in.set_index(['img'], inplace=True)

df_in = df_in.filter(regex='x_')

from sklearn.model_selection import train_test_split

df_train_split, df_valid_split = train_test_split(df_in, test_size=0.2, random_state=42)

df_in.head()
import cv2



def bbox(img):

    img = (img > 0)

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))

    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))

    return rmin, rmax, cmin, cmax



def read_mask_centered(name):

    png_file = '{}/train_masks/{}_mask.png'.format(INPUT_PATH, name)

    if os.path.exists(png_file):

        img_arr = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)

    else:

        gif_file = '{}/train_masks/{}_mask.gif'.format(INPUT_PATH, name)

        from scipy import ndimage

        img_arr = ndimage.imread(gif_file,flatten=True)



    # find mask bounding box

    r1, r2, c1, c2 = bbox(img_arr)

    # crop and resize

    img_cr = img_arr[r1:r2, c1:c2]

    mask = cv2.resize(img_cr,(IMG_SIZE,IMG_SIZE))

    return mask

def input_generator(df_in, batch_size):

    while True:

        for start in range(0, len(df_in), batch_size):

            end = min(start + batch_size, len(df_in))

            x_batch = np.array(df_in[start:end], dtype=np.float32)

            y_batch = []

            for img_id in df_in[start:end].index:

                mask = read_mask_centered(img_id)

                mask = np.expand_dims(mask, axis=2)

                y_batch.append(mask)

            y_batch = np.array(y_batch, np.float32) / 255

            yield x_batch, y_batch

from keras.layers import Input, concatenate, Conv2D, UpSampling2D, BatchNormalization, Reshape, Dropout

from keras.losses import binary_crossentropy

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras.backend as K



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))
def create_up(input, filter_count, n_layers=2):

    up = UpSampling2D((2, 2))(input)

    up = BatchNormalization()(up)

    up = Dropout(0.7)(up)

    for i in range(n_layers):

        up = Conv2D(filter_count, (3, 3), padding='same', activation='relu')(up)

        up = BatchNormalization()(up)

    return up



def create_model(n_inputs):

    inputs = Input(shape=(n_inputs,))

    x = Reshape((1,1,n_inputs))(inputs)

    n_filters = n_inputs // 2

    for depth in range(IMG_SIZE_LOG2):

        x = create_up(x, n_filters)

        n_filters = int(n_filters / 1.5)

    classify = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=classify) 

    return model



n_inputs = df_in.values.shape[1]

model = create_model(n_inputs)

#model.summary(line_length=80)

model.compile(optimizer=Adam(), loss=bce_dice_loss, metrics=[dice_loss])

batch_size = 16

epochs=2

model.fit_generator(generator=input_generator(df_train_split,batch_size),

                    steps_per_epoch=np.ceil(float(len(df_train_split)) / float(batch_size)),

                    epochs=epochs,

                    verbose=1,

                    workers=1,

                    max_queue_size=2*batch_size,

                    validation_data=input_generator(df_valid_split,batch_size),

                    validation_steps=np.ceil(float(len(df_valid_split)) / float(batch_size)))

import matplotlib.pylab as plt

plt.figure(figsize=(20, 20))



for i,batch in enumerate(input_generator(df_valid_split,1)):

    if i >= 9:

        break

    id=df_valid_split.index[i]   

    x_batch, y_true = batch

    y_batch = model.predict(x_batch)



    plt.subplot(9,2,2*i+1)

    plt.title('%s true'%id)

    plt.axis('off')

    plt.imshow(y_true[0,:,:,0])

    plt.subplot(9,2,2*i+2)

    plt.title('%s predicted'%id)

    plt.axis('off')

    plt.imshow(y_batch[0,:,:,0])    