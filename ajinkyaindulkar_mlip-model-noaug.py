# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load 



import warnings

warnings.filterwarnings(action='ignore')



import cv2 # image processing

import time, gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from tqdm.auto import tqdm

import albumentations as A



# deep learning frameworks

import tensorflow as tf

from tensorflow import keras

from keras.models import Model

from keras.optimizers import Adam

from keras.models import clone_model

from keras.utils.vis_utils import plot_model

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input,Reshape,Concatenate, concatenate,GaussianDropout 





# data visualization/serialization packages

import pickle

import seaborn as sns

import matplotlib.image as mpimg

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df_.head()


test_df_.head()
sample_sub_df.head()
class_map_df.head()
print(f'Size of training data: {train_df_.shape}')

print(f'Size of test data: {test_df_.shape}')

print(f'Size of class map: {class_map_df.shape}')
HEIGHT = 236

WIDTH = 236



def get_n(df, field, n, top=True):

    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]

    top_grapheme_roots = top_graphemes.index

    top_grapheme_counts = top_graphemes.values

    top_graphemes = class_map_df.iloc[top_grapheme_roots]

    top_graphemes.drop(['component_type', 'label'], axis=1, inplace=True)

    top_graphemes.loc[:, 'count'] = top_grapheme_counts

    return top_graphemes



def image_from_char(char):

    image = Image.new('RGB', (WIDTH, HEIGHT))

    draw = ImageDraw.Draw(image)

    myfont = ImageFont.truetype('/kaggle/input/banglafonts/SolaimanLipi.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)



    return image
print(f'Number of unique grapheme roots: {train_df_["grapheme_root"].nunique()}')

print(f'Number of unique vowel diacritic: {train_df_["vowel_diacritic"].nunique()}')

print(f'Number of unique consonant diacritic: {train_df_["consonant_diacritic"].nunique()}')
top_10_roots = get_n(train_df_, 'grapheme_root', 10)

top_10_roots
f, ax = plt.subplots(2, 5, figsize=(10, 5))

ax = ax.flatten()



for i in range(10):

    ax[i].imshow(image_from_char(top_10_roots['component'].iloc[i]), cmap='Greys')
bottom_10_roots = get_n(train_df_, 'grapheme_root', 10, False)

bottom_10_roots
f, ax = plt.subplots(2, 5, figsize=(10, 5))

ax = ax.flatten()



for i in range(10):

    ax[i].imshow(image_from_char(bottom_10_roots['component'].iloc[i]), cmap='Greys')
top_5_vowels = get_n(train_df_, 'vowel_diacritic', 5)

top_5_vowels
f, ax = plt.subplots(1, 5, figsize=(10, 5))

ax = ax.flatten()



for i in range(5):

    ax[i].imshow(image_from_char(top_5_vowels['component'].iloc[i]), cmap='Greys')
top_5_consonants = get_n(train_df_, 'consonant_diacritic', 5)

top_5_consonants
f, ax = plt.subplots(1, 5, figsize=(10, 5))

ax = ax.flatten()



for i in range(5):

    ax[i].imshow(image_from_char(top_5_consonants['component'].iloc[i]), cmap='Greys')
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)
train_df_str = train_df_.applymap(str)

train_df_str['identifyer'] = train_df_str['grapheme_root']+','+train_df_str['vowel_diacritic']+','+train_df_str['consonant_diacritic']



train_df_str
observed = train_df_str.groupby(['identifyer']).size() #.sort_values(by=['col1'])

observed.plot.hist(bins=12)

observed.sort_values().head(10)
# memory clean-up

del top_10_roots, bottom_10_roots

del top_5_vowels, top_5_consonants

del train_df_str, observed

del f, ax
train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
IMG_SIZE=64

N_CHANNELS=1
HEIGHT = 137

WIDTH = 236

SIZE = 64

CROP_SIZE = 64
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    resize_size=64

    angle=0

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            #Reshape

            image=df.loc[df.index[i]].values.reshape(137,236)

            

            #Centering

            image_center = tuple(np.array(image.shape[1::-1]) / 2)

            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Scaling

            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Brightness

#             augBright=A.RandomBrightnessContrast(p=1.0)

#             image = augBright(image=image)['image']

            

            #Threshold and Contours

            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)



            resized[df.index[i]] = resized_roi.reshape(-1)

        

    else:

        for i in range(df.shape[0]):

            #Reshape

            image=df.loc[df.index[i]].values.reshape(137,236)

            

            #Centering

            image_center = tuple(np.array(image.shape[1::-1]) / 2)

            matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Scaling

            matrix = cv2.getRotationMatrix2D(image_center, 0, 1.0)

            image = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,

                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            

            #Brightness

#             augBright=A.RandomBrightnessContrast(p=1.0)

#             image = augBright(image=image)['image']

            

            #Threshold and Contours

            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]



            idx = 0 

            ls_xmin = []

            ls_ymin = []

            ls_xmax = []

            ls_ymax = []

            for cnt in contours:

                idx += 1

                x,y,w,h = cv2.boundingRect(cnt)

                ls_xmin.append(x)

                ls_ymin.append(y)

                ls_xmax.append(x + w)

                ls_ymax.append(y + h)

            xmin = min(ls_xmin)

            ymin = min(ls_ymin)

            xmax = max(ls_xmax)

            ymax = max(ls_ymax)



            roi = image[ymin:ymax,xmin:xmax]

            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)



            resized[df.index[i]] = resized_roi.reshape(-1)

            

    resized = pd.DataFrame(resized).T

    return resized
from keras.applications import Xception



# load the model

inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

model = Conv2D(filters=3, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)

model = BatchNormalization(momentum=0.15)(model)

model = GaussianDropout(rate=0.3)(model)

fmodel = Conv2D(filters=3, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

fmodel = BatchNormalization(momentum=0.15)(fmodel)



# Pretrained model

base_model = Xception(weights=None, include_top=False)(fmodel)



model = BatchNormalization(momentum=0.15)(base_model)

model = GaussianDropout(rate=0.3)(model)

model = Flatten()(base_model)



model = Dense(1024, activation = "relu")(model)

model = BatchNormalization(momentum=0.15)(model)

model = GaussianDropout(rate=0.3)(model)

dense = Dense(512, activation = "relu")(model)

dense = BatchNormalization(momentum=0.15)(dense)

head_root = Dense(168, activation = 'softmax')(dense)



head_root = Dense(168, activation = 'softmax')(dense)

head_vowel = Dense(11, activation = 'softmax')(dense)

head_consonant = Dense(7, activation = 'softmax')(dense)



model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

    

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.layers[13].trainable=True # We freeze redundant characteristics



    

model.layers[14].trainable=True

model.layers[15].trainable=True



# model.layers[7].trainable=True

for i,layer in enumerate(model.layers):

     print(i,layer.name,layer.trainable)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_4_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.000001)

learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_5_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.3, 

                                            min_lr=0.000001)

learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_6_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.000001)
#decreased batch size

batch_size = 128

epochs = 30
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):



    def flow(self,

             x,

             y=None,

             batch_size=32,

             shuffle=True,

             sample_weight=None,

             seed=None,

             save_to_dir=None,

             save_prefix='',

             save_format='png',

             subset=None):



        targets = None

        target_lengths = {}

        ordered_outputs = []

        for output, target in y.items():

            if targets is None:

                targets = target

            else:

                targets = np.concatenate((targets, target), axis=1)

            target_lengths[output] = target.shape[1]

            ordered_outputs.append(output)





        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,

                                         shuffle=shuffle):

            target_dict = {}

            i = 0

            for output in ordered_outputs:

                target_length = target_lengths[output]

                target_dict[output] = flowy[:, i: i + target_length]

                i += target_length



            yield flowx, target_dict
histories = []

for i in range(4):

    model.save_weights("model_noaug_base.h5")

    

    # load data

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)

    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

    X_train = resize(X_train)/255

    

    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images

    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    # prepare data

    Y_train_root = pd.get_dummies(train_df['grapheme_root']).values

    Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values

    Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values



    print(f'Training images: {X_train.shape}')

    print(f'Training labels root: {Y_train_root.shape}')

    print(f'Training labels vowel: {Y_train_vowel.shape}')

    print(f'Training labels consonants: {Y_train_consonant.shape}')



    # Divide the data into training and validation set

    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, 

                                                                                                                                    Y_train_vowel, 

                                                                                                                                    Y_train_consonant, 

                                                                                                                                    test_size=0.08, 

                                                                                                                                    random_state=666)

    # memory clean-up

    del train_df

    del X_train

    del Y_train_root, Y_train_vowel, Y_train_consonant



    # Data augmentation for creating more training data - NO AUGMENTATION

    datagen = MultiOutputDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.0, # Randomly zoom image 

        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





    # This will just calculate parameters required to augment the given data. This won't perform any augmentations

    datagen.fit(x_train)



    # Fit the model

    history = model.fit_generator(datagen.flow(x_train, {'dense_4': y_train_root, 'dense_5': y_train_vowel, 'dense_6': y_train_consonant}, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

                              steps_per_epoch=x_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant])



    histories.append(history)

    

    # memory clean-up

    del x_train

    del x_test

    del y_train_root

    del y_test_root

    del y_train_vowel

    del y_test_vowel

    del y_train_consonant

    del y_test_consonant

    gc.collect()
%matplotlib inline

def plot_loss(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_4_loss'], label='train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_5_loss'], label='train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['dense_6_loss'], label='train_consonant_loss')

    

    plt.plot(np.arange(0, epoch), his.history['val_dense_4_loss'], label='val_train_root_loss')

    plt.plot(np.arange(0, epoch), his.history['val_dense_5_loss'], label='val_train_vowel_loss')

    plt.plot(np.arange(0, epoch), his.history['val_dense_6_loss'], label='val_train_consonant_loss')

    

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['dense_4_accuracy'], label='train_root_acc')

    plt.plot(np.arange(0, epoch), his.history['dense_5_accuracy'], label='train_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['dense_6_accuracy'], label='train_consonant_accuracy')

    

    plt.plot(np.arange(0, epoch), his.history['val_dense_4_accuracy'], label='val_root_acc')

    plt.plot(np.arange(0, epoch), his.history['val_dense_5_accuracy'], label='val_vowel_accuracy')

    plt.plot(np.arange(0, epoch), his.history['val_dense_6_accuracy'], label='val_consonant_accuracy')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
for dataset in range(4):

    plot_loss(histories[dataset], epochs, f'Training Dataset: {dataset}')

    plot_acc(histories[dataset], epochs, f'Training Dataset: {dataset}')
model_name = "model_noaug_v1.h5"

model.save_weights(model_name)



with open('histRoot_noaug_v1.pkl', 'wb') as fp:

    pickle.dump(histories, fp)