# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2



import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
test_df_.head()
sample_sub_df.head(10)
class_map_df.head(10)
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

    myfont = ImageFont.truetype('/kaggle/input/bengali-fonts/hind_siliguri_normal_500.ttf', 120)

    w, h = draw.textsize(char, font=myfont)

    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)



    return image
print(f'Number of unique grapheme roots: {train_df_["grapheme_root"].nunique()}')

print(f'Number of unique vowel diacritic: {train_df_["vowel_diacritic"].nunique()}')

print(f'Number of unique consonant diacritic: {train_df_["consonant_diacritic"].nunique()}')
top_10_roots = get_n(train_df_, 'grapheme_root', 10)

top_10_roots
f, ax = plt.subplots(2, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(10):

    ax[i].imshow(image_from_char(top_10_roots['component'].iloc[i]), cmap='Greys')
bottom_10_roots = get_n(train_df_, 'grapheme_root', 10, False)

bottom_10_roots
f, ax = plt.subplots(2, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(10):

    ax[i].imshow(image_from_char(bottom_10_roots['component'].iloc[i]), cmap='Greys')
top_5_vowels = get_n(train_df_, 'vowel_diacritic', 5)

top_5_vowels
f, ax = plt.subplots(1, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(5):

    ax[i].imshow(image_from_char(top_5_vowels['component'].iloc[i]), cmap='Greys')
top_5_consonants = get_n(train_df_, 'consonant_diacritic', 5)

top_5_consonants
f, ax = plt.subplots(1, 5, figsize=(16, 8))

ax = ax.flatten()



for i in range(5):

    ax[i].imshow(image_from_char(top_5_consonants['component'].iloc[i]), cmap='Greys')
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)
train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
IMG_SIZE=64

N_CHANNELS=1
def resize(df, size=64, need_progress_bar=True):

    resized = {}

    if need_progress_bar:

        for i in tqdm(range(df.shape[0])):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

    else:

        for i in range(df.shape[0]):

            image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size))

            resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T

    return resized
def get_dummies(df):

    cols = []

    for col in df:

        cols.append(pd.get_dummies(df[col].astype(str)))

    return pd.concat(cols, axis=1)
model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(64, 64, 1)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu'))

model.add(Dropout(rate=0.3))



model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu'))

model.add(Dropout(rate=0.3))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.40))

model.add(Dense(192, activation = "relu"))

model.add(Dropout(0.40))

# model.add(Dense(186, activation = "softmax"))
model_root = clone_model(model)

model_vowel = clone_model(model)

model_consonant = clone_model(model)
model_root.add(Dense(168, activation = 'softmax'))

model_vowel.add(Dense(11, activation = 'softmax'))

model_consonant.add(Dense(7, activation = 'softmax'))
model_root.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

model_vowel.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])

model_consonant.compile(optimizer="adam", loss=['categorical_crossentropy'], metrics=['accuracy'])
model_root.summary()
model_vowel.summary()
model_consonant.summary()
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1,

                                            factor=0.5, 

                                            min_lr=0.00001)
batch_size = 64

epochs = 12
model_dict = {

    'grapheme_root': model_root,

    'vowel_diacritic': model_vowel,

    'consonant_diacritic': model_consonant

}
histories = []

for i in range(4):

    train_df = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df_, on='image_id').drop(['image_id'], axis=1)

    

    # Visualize few samples of current training dataset

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

    count=0

    for row in ax:

        for col in row:

            col.imshow(resize(train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(64, 64))

            count += 1

    plt.show()

    

    X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)

    X_train = resize(X_train)/255

    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images

    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    

    for target in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:

        Y_train = train_df[target]

        Y_train = pd.get_dummies(Y_train).values



        print(f'Training images: {X_train.shape}')

        print(f'Training labels: {Y_train.shape}')



        # Divide the data into training and validation set

        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.15, random_state=666)

        del Y_train

        

        # Data augmentation for creating more training data

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

        

         # Fit the model

        history = model_dict[target].fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                                      epochs = epochs, validation_data = (x_test,y_test),

                                      steps_per_epoch=x_train.shape[0] // batch_size, 

                                      callbacks=[learning_rate_reduction])

    

        histories.append(history)

        del x_train

        del x_test

        del y_train

        del y_test

        gc.collect()

    # Delete to reduce memory usage

    del X_train

    del train_df

    gc.collect()
%matplotlib inline

def plot_loss(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['val_loss'], label='val_loss')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Loss')

    plt.legend(loc='upper right')

    plt.show()



def plot_acc(his, epoch, title):

    plt.style.use('ggplot')

    plt.figure()

    plt.plot(np.arange(0, epoch), his.history['accuracy'], label='train_acc')

    plt.plot(np.arange(0, epoch), his.history['val_accuracy'], label='val_accuracy')

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Accuracy')

    plt.legend(loc='upper right')

    plt.show()
for dataset in range(4):

    for target in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:

        plot_loss(histories[0], epochs, f'Dataset: {dataset}, Training on: {target}')

        plot_acc(histories[0], epochs, f'Dataset: {dataset}, Training on: {target}')
del histories

del model

gc.collect()
preds_dict = {

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

    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



    for pred in preds_dict:

        preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)



    for k,id in enumerate(df_test_img.index.values):  

        for i,comp in enumerate(components):

            id_sample=id+'_'+comp

            row_id.append(id_sample)

            target.append(preds_dict[comp][k])

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
# components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

# target=[] # model predictions placeholder

# row_id=[] # row_id place holder

# n_cls = [7,168,11] # number of classes in each of the 3 targets

# for i in range(4):

#     df_test_img = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 

#     df_test_img.set_index('image_id', inplace=True)



#     X_test = resize(df_test_img)/255

#     X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)



#     for pred in preds_dict:

#         preds_dict[pred]=np.argmax(model_dict[pred].predict(X_test), axis=1)



#     for k,id in enumerate(df_test_img.index.values):  

#         for i,comp in enumerate(components):

#             id_sample=id+'_'+comp

#             row_id.append(id_sample)

#             target.append(preds_dict[comp][k])



# df_sample = pd.DataFrame(

#     {'row_id': row_id,

#     'target':target

#     },

#     columns =['row_id','target'] 

# )

# df_sample.to_csv('submission.csv',index=False)