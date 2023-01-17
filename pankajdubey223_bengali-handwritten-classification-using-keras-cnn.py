# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)













import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2

















# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')

train_df.head()
test_df.head()
class_map_df.head()
sample_df.head()
print('shape of the train_df is : ',train_df.shape)

print('shape of the test_df is : ',test_df.shape)

print('shape of the class_map_df is : ',class_map_df.shape)

print('shape of the sample_df is : ',sample_df.shape)
print(f'Number of unique grapheme roots: {train_df["grapheme_root"].nunique()}')

print(f'Number of unique vowel diacritic: {train_df["vowel_diacritic"].nunique()}')

print(f'Number of unique consonant diacritic: {train_df["consonant_diacritic"].nunique()}')

train_df.groupby('grapheme')['vowel_diacritic'].agg('sum').head()
for i in range(5):

    print(train_df['image_id'][i])
train_df = train_df.drop(['grapheme'], axis=1, inplace=False)

train_df
train_df.dtypes
train_df.dtypes
from keras.models  import Sequential

from keras.models import Model

from keras.layers import Conv2D

from keras.layers import MaxPool2D,Input

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import BatchNormalization
IMG_SIZE = 64


inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))



model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.3)(model)



model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = MaxPool2D(pool_size=(2, 2))(model)

model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)

model = BatchNormalization(momentum=0.15)(model)

model = Dropout(rate=0.3)(model)

model = Flatten()(model)

model = Dense(1024, activation = "relu")(model)

model = Dropout(rate=0.3)(model)

dense = Dense(512, activation = "relu")(model)

head_root = Dense(168, activation = 'softmax')(dense)

head_vowel = Dense(11, activation = 'softmax')(dense)

head_consonant = Dense(7, activation = 'softmax')(dense)



model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_df.shape
import matplotlib.pyplot as plt

histories = []

for i in range(4):

    train_df_ = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df, on='image_id').drop(['image_id'], axis=1)

    

    # Visualize few samples of current training dataset

   # fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

    #count=0

    #for row in ax:

       # for col in row:

        #    col.imshow(resize(train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))

         #   count += 1

    #plt.show()

    

    #X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

    #X_train = resize(X_train)/255

    

    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images

    #X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    
train_df_.shape
X_train = train_df_.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

X_train.shape
len(X_train)
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    resize_size=64

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image=df.loc[df.index[i]].values.reshape(137,236)

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

            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

            image=df.loc[df.index[i]].values.reshape(137,236)

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
X_train = resize(X_train)/255

X_train.shape
X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(X_train.shape)

Y_train_root = pd.get_dummies(train_df_['grapheme_root']).values

Y_train_vowel = pd.get_dummies(train_df_['vowel_diacritic']).values

Y_train_consonant = pd.get_dummies(train_df_['consonant_diacritic']).values



print(f'Training images: {X_train.shape}')

print(f'Training labels root: {Y_train_root.shape}')

print(f'Training labels vowel: {Y_train_vowel.shape}')

print(f'Training labels consonants: {Y_train_consonant.shape}')

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)



del train_df

del X_train

del Y_train_root, Y_train_vowel, Y_train_consonant

from tensorflow import keras

class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):



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
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





    # This will just calculate parameters required to augment the given data. This won't perform any augmentations

datagen.fit(x_train)



import gc

gc.collect()
classifier = []

batch_size = 256

epochs = 50







history = model.fit_generator(datagen.flow(x_train, {'dense_3': y_train_root, 'dense_4': y_train_vowel, 'dense_5': y_train_consonant}, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 

                              steps_per_epoch=x_train.shape[0] // batch_size )

 

histories.append(history)



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

def plot_loss(cls,epoch,title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0,epoch),cls.history['loss'],label = 'train_loss')

    plt.plot(np.arange(0,epoch),cls.history['dense_3_loss'],label = 'train_root_loss')

    plt.plot(np.arange(0,epoch),cls.history['dense_4_loss'],label = 'train_vowel_loss')

    plt.plot(np.arange(0,epoch),cls.history['dense_5_loss'],label = 'train_consonant_loss')

    plt.plot(np.arange(0,epoch),cls.history['val_dense_3_loss'],label = 'val_train_root_loss')

    plt.plot(np.arange(0,epoch),cls.history['val_dense_4_loss'],label = 'val_train_vowel_loss')

    plt.plot(np.arange(0,epoch),cls.history['val_dense_5_loss'],label = 'val_train_consonant_loss')

    plt.title(title)

    plt.legend(loc='upper right')

    plt.xlabel('Number of Epoch ')

    plt.ylabel('Loss')

    plt.show()

    

    

    

def plot_accuracy(cls,epoch,title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0,epoch),cls.history['dense_3_accuracy'],label = 'train_root_accuracy')

    plt.plot(np.arange(0,epoch),cls.history['dense_4_accuracy'],label = 'train_vowel_accuracy')

    plt.plot(np.arange(0,epoch),cls.history['dense_5_accuracy'],label = 'train_comnsonant_accuracy')

    plt.plot(np.arange(0,epoch),cls.history['val_dense_3_accuracy'],label = 'val_train_root_accuracy')

    plt.plot(np.arange(0,epoch),cls.history['val_dense_4_accuracy'],label = 'val_train_vowel_accuracy')

    plt.plot(np.arange(0,epoch),cls.history['val_dense_5_accuracy'],label = 'val_train_consonant_accuracy')

    plt.title(title)

    plt.xlabel('Number of Epoch')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
for dataset in range(1):

    plot_loss(histories[dataset], epochs, f'Training Dataset: {dataset}')

    plot_accuracy(histories[dataset], epochs, f'Training Dataset: {dataset}')
pred_dict = {

    'grapheme_root': [],

    'vowel_diacritic': [],

    'consonant_diacritic': []

}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target=[] # model predictions placeholder

row_id=[] # row_id place holder

for i in range(4):

    df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

    df_test_img.set_index('image_id', inplace=True)



    X_test = resize(df_test_img, need_progress_bar=False)/255

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    

    preds = model.predict(X_test)



    for i, p in enumerate(pred_dict):

        pred_dict[p] = np.argmax(preds[i], axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(pred_dict[comp][k])

    del df_test_img

    del X_test

    gc.collect()



df_sample = pd.DataFrame(

    {

        'row_id': row_id,

        'target':target

    },

    columns = ['row_id','target'] 

)

df_sample.to_csv('submission.csv',index=False)

df_sample.head()