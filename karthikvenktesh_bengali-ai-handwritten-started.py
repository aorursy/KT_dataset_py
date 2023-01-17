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
import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, Activation, MaxPooling2D, Input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
ORIGINAL_IMG_HEIGHT = 137

ORIGINAL_IMG_WIDTH = 236

IMG_SIZE = 128

BASE_PATH='/kaggle/input/bengaliai-cv19/'

EPOCHS = 15

GRAPHME_CLASSES = 168

VOWEL_CLASSES = 11

CONSONENT_CLASSES = 7
class_map_ds = pd.read_csv(BASE_PATH + 'class_map.csv')

class_map_ds.head(5)
test_df = pd.read_csv(BASE_PATH + 'test.csv')

test_df.head(5)
train_df = pd.read_csv(BASE_PATH + 'train.csv')

train_df.head(5)
train_image_files = ['train_image_data_0.parquet', 'train_image_data_1.parquet', 'train_image_data_2.parquet', 'train_image_data_3.parquet']
def show_batch(images_array, labels_array = None, resize_shape = None):

  plt.figure(figsize=(10,10))

  for n in range(25):

    ax = plt.subplot(5,5,n+1)

    img = images_array[n]

    if resize_shape is not None:

        img = img.reshape(resize_shape)

    plt.imshow(img)

    if labels_array is not None:

        plt.title(labels_array[n])

    plt.axis('off')
def show_accuracy(history):

    grapheme_op_acc = history.history['grapheme_op_accuracy']

    vowel_op_acc = history.history['vowel_op_accuracy']

    consonent_op_acc = history.history['consonent_op_accuracy']



    grapheme_op_loss = history.history['grapheme_op_loss']

    vowel_op_loss = history.history['vowel_op_loss']

    consonent_op_loss = history.history['consonent_op_loss']

    loss = history.history['loss']



    val_grapheme_op_acc = history.history['val_grapheme_op_accuracy']

    val_vowel_op_acc = history.history['val_vowel_op_accuracy']

    val_consonent_op_acc = history.history['val_consonent_op_accuracy']



    val_grapheme_op_loss = history.history['val_grapheme_op_loss']

    val_vowel_op_loss = history.history['val_vowel_op_loss']

    val_consonent_op_loss = history.history['val_consonent_op_loss']

    val_loss = history.history['val_loss']



    epochs_range = range(EPOCHS)



    plt.figure(figsize=(20, 5))

    

    plt.subplot(1, 4, 1)

    plt.plot(epochs_range, grapheme_op_acc, label='Grapheme accuracy')

    plt.plot(epochs_range, vowel_op_acc, label='Vowel Accuracy')

    plt.plot(epochs_range, consonent_op_acc, label='Consonent accuracy')

    plt.legend(loc='lower right')

    plt.title('Training Accuracy')



    plt.subplot(1, 4, 2)

    plt.plot(epochs_range, val_grapheme_op_acc, label='Val grapheme accuracy')

    plt.plot(epochs_range, val_vowel_op_acc, label='Val vowel Accuracy')

    plt.plot(epochs_range, val_consonent_op_acc, label='Val consonent accuracy')

    plt.legend(loc='lower right')

    plt.title('Validation Accuracy')



    plt.subplot(1, 4, 3)

    plt.plot(epochs_range, val_grapheme_op_acc, label='Grapheme loss')

    plt.plot(epochs_range, val_vowel_op_acc, label='Vowel loss')

    plt.plot(epochs_range, val_consonent_op_acc, label='Consonent loss')

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.legend(loc='upper right')

    plt.title('Training Loss')



    plt.subplot(1, 4, 4)

    plt.plot(epochs_range, val_grapheme_op_loss, label='Val grapheme loss')

    plt.plot(epochs_range, val_vowel_op_loss, label='Val vowel loss')

    plt.plot(epochs_range, val_consonent_op_loss, label='Val consonent loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.title('Validation Loss')



    plt.show()
def split(df):

    non_image_coiumn = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

    X_train, X_val, y_train, y_val =  train_test_split(

        df.drop(non_image_coiumn, axis=1), 

        df.loc[:, non_image_coiumn],

        test_size=0.25, 

        random_state=42

      )

    X_val, X_test, y_val, y_test =  train_test_split(

        X_val, 

        y_val,

        test_size=0.2, 

        random_state=42

      )

    return X_train, X_val, X_test, y_train, y_val, y_test
def input_pipeline(file_name):

    df = pd.merge(

            pd.read_parquet(BASE_PATH + file_name), 

            train_df,

            on='image_id'

          ).drop(['image_id', 'grapheme'], axis=1)



    return split(df=df)
def resize_images(images):

    return [cv2.resize(image, (IMG_SIZE, IMG_SIZE)) for image in images]
def get_train_test_data(file_name):

    X_train, X_val, X_test, y_train, y_val, y_test = input_pipeline(file_name)

    X_train = X_train.to_numpy(dtype='f').reshape(X_train.shape[0],ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)

    X_val = X_val.to_numpy(dtype='f').reshape(X_val.shape[0],ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)

    X_test = X_test.to_numpy(dtype='f').reshape(X_test.shape[0],ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)

    X_train = resize_images(X_train)

    X_val = resize_images(X_val)

    X_test = resize_images(X_test)

    y_train = y_train.to_numpy()

    y_val = y_val.to_numpy()

    y_test = y_test.to_numpy()

    return X_train, X_val, X_test, y_train, y_val, y_test
inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = Conv2D(16, 3, padding='same', activation='relu')(inp)

x = MaxPooling2D()(x)

x = Dropout(0.2)(x)

x = Conv2D(32, 3, padding='same', activation='relu')(x)

x = MaxPooling2D()(x)

x = Conv2D(64, 3, padding='same', activation='relu')(x)

x = MaxPooling2D()(x)

x = Dropout(0.2)(x)

x = Flatten()(x)





grapme_op = Dense(GRAPHME_CLASSES, activation='softmax', name='grapheme_op')(x)

vowel_op = Dense(VOWEL_CLASSES, activation='softmax', name='vowel_op')(x)

consonent_op = Dense(CONSONENT_CLASSES, activation='softmax', name='consonent_op')(x)



model = Model(inputs=inp, outputs=[grapme_op, vowel_op, consonent_op])
losses = {

	'grapheme_op': 'sparse_categorical_crossentropy',

	'vowel_op': 'sparse_categorical_crossentropy',

    'consonent_op': 'sparse_categorical_crossentropy'

}

lossWeights = {

    'grapheme_op': 1.0, 

    'vowel_op': 1.0, 

    'consonent_op': 1.0

}

metrices = {

    'grapheme_op': 'accuracy', 

    'vowel_op': 'accuracy', 

    'consonent_op': 'accuracy'

}
model.compile(optimizer='adam',

              loss=losses,

              loss_weights=lossWeights,

              metrics=metrices)
model.summary()
histories = []



y_test_graphemes = []

y_test_vowels = []

y_test_consonents = []



pred_graphemes = []

pred_vowels = []

pred_consonents = []
def index_of_max(arr):

    ind = np.where(arr == np.max(arr))

    return ind[0][0]
for file in train_image_files:

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_data(file_name=file)



    X_train = np.expand_dims(X_train, axis=-1)

    X_val = np.expand_dims(X_val, axis=-1)

    X_test = np.expand_dims(X_test, axis=-1)

    

    y_train_grapheme = y_train[:, 0]

    y_train_vowel = y_train[:, 1]

    y_train_consonent = y_train[:, 2]



    y_val_grapheme = y_val[:, 0]

    y_val_vowel = y_val[:, 1]

    y_val_consonent = y_val[:, 2]



    train_op = {

        'grapheme_op': y_train_grapheme, 

        'vowel_op': y_train_vowel, 

        'consonent_op': y_train_consonent

    }



    val_op = {

        'grapheme_op': y_val_grapheme, 

        'vowel_op': y_val_vowel, 

        'consonent_op': y_val_consonent

    }



    history = model.fit(

          X_train,

          train_op,

          validation_data=(X_val, val_op),

          epochs=EPOCHS

        )



    y_test_grapheme = y_test[:, 0]

    y_test_vowel = y_test[:, 1]

    y_test_consonent = y_test[:, 2]



    pred = model.predict(X_test)

    pred_grapheme = [index_of_max(item) for item in pred[0]]

    pred_vowel = [index_of_max(item) for item in pred[1]]

    pred_consonent = [index_of_max(item) for item in pred[2]]



    histories.append(history)

    

    y_test_graphemes.append(y_test_grapheme)

    y_test_vowels.append(y_test_vowel)

    y_test_consonents.append(y_test_consonent)



    pred_graphemes.append(pred_grapheme)

    pred_vowels.append(pred_vowel)

    pred_consonents.append(pred_consonent)

      

    del X_train

    del X_val

    del y_train

    del y_val

    del y_train_grapheme

    del y_train_vowel 

    del y_train_consonent

    del y_val_grapheme

    del y_val_vowel 

    del y_val_consonent
for history in histories:

  show_accuracy(history)
for i in range(len(train_image_files)):

  print('After trainining with : ', train_image_files[i])



  true_labels = y_test_graphemes[i]

  predict_labels = pred_graphemes[i]

  print('Grapheme Accuracy Score :',accuracy_score(true_labels, predict_labels))



  true_labels = y_test_vowels[i]

  predict_labels = pred_vowels[i]

  print('Vowel Accuracy Score :',accuracy_score(true_labels, predict_labels))



  true_labels = y_test_consonents[i]

  predict_labels = pred_consonents[i]

  print('Consonent Accuracy Score :',accuracy_score(true_labels, predict_labels))



  print('********************************************************************')
test_image_files = ['test_image_data_0.parquet', 'test_image_data_1.parquet', 'test_image_data_2.parquet', 'test_image_data_3.parquet']
test_image_df = None



for files in test_image_files:

  df = pd.read_parquet(BASE_PATH + files)

  if test_image_df is None:

    test_image_df = df

  else:

    test_image_df = pd.concat([test_image_df, df], ignore_index=True)

  

  del df



test_image_df.head()
test_image_df = test_image_df.drop(['image_id'], axis = 'columns')

test_image_df.head()
test_images = test_image_df.to_numpy(dtype='f').reshape(test_image_df.shape[0],ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH)

test_images = resize_images(test_images)

test_images = np.expand_dims(test_images, axis=-1)
pred = model.predict(test_images)

pred_grapheme = [index_of_max(item) for item in pred[0]]

pred_vowel = [index_of_max(item) for item in pred[1]]

pred_consonent = [index_of_max(item) for item in pred[2]]
result = []

for i in range(len(test_images)):

  result.append(pred_grapheme[i])

  result.append(pred_vowel[i])

  result.append(pred_consonent[i])
test_df.head()
test_df.drop(['component'], axis=1)

test_df['component'] = result 

test_df.head()