import gc

import os

from os import listdir

from os.path import isfile, join



import zipfile

import itertools



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import cv2



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras.metrics import AUC, Precision, Recall

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Extracting zip files



with zipfile.ZipFile("../input/dogs-vs-cats-redux-kernels-edition/train.zip","r") as z:

     z.extractall("../working/dogs-vs-cats/train")

    

with zipfile.ZipFile("../input/dogs-vs-cats-redux-kernels-edition/test.zip","r") as z:

    z.extractall("../working/dogs-vs-cats/test")
print(os.listdir('../working/dogs-vs-cats/'))
# Cat = 0, Dog = 1

def get_dataset():

    path = './dogs-vs-cats/train/train'

    dataset = []

    for f in listdir(path):

        if isfile(join(path, f)):

            if 'cat' in f:

                dataset.append({'label': 0, 'img_path': join(path, f)})

            elif 'dog' in f:

                dataset.append({'label': 1, 'img_path': join(path, f)})

    return dataset





# Getting the training dataset

dataset = get_dataset()
df = pd.DataFrame(dataset)



# shuffling the dataframe and resetting the index

df = df.sample(frac=1).reset_index(drop=True)



df.head()
# Splitting the training and development sets

X_train, X_dev, Y_train, Y_dev = train_test_split(df.img_path, df.label, test_size=0.1, random_state=0)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)



# Converting pandas series to dataframe

train_df = pd.DataFrame({ 'label': Y_train, 'img_path': X_train }).reset_index(drop=True)

dev_df   = pd.DataFrame({ 'label': Y_dev, 'img_path': X_dev }).reset_index(drop=True)

test_df  = pd.DataFrame({ 'label': Y_test, 'img_path': X_test }).reset_index(drop=True)



train_df.head()
# See if training set is balanced or not

sns.countplot(train_df.label)
# Looking at first 25 training examples



plt.figure(figsize=(15, 15))

for i, _img_path in enumerate(X_train[:25]):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    img = load_img(_img_path)

    plt.imshow(img)
# Constants



IMG_HEIGHT = 150

IMG_WIDTH  = 150

CHANNELS   = 3



BATCH_SIZE = 16
# Used for processing test data (./dogs-vs-cats/test/test -- directory)

def process_images(list_of_images):

    x = []  # holds images

    y = []  # hold labels

    

    for image in list_of_images:

        x.append(

            cv2.resize(

                cv2.imread(image, cv2.IMREAD_COLOR),

                (IMG_HEIGHT, IMG_WIDTH),

                interpolation=cv2.INTER_CUBIC

            )

        )

        

        if 'dog' in image:

            y.append(1)

        if 'cat' in image:

            y.append(0)

    

    return x, y
train_datagen = ImageDataGenerator(

    rescale=1/255, 

    rotation_range=10, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    horizontal_flip=True,

)

dev_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)





train_generator = train_datagen.flow_from_dataframe(

    train_df,

    x_col='img_path',

    y_col='label',

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    shuffle=True,

    class_mode='raw'

)

dev_generator = dev_datagen.flow_from_dataframe(

    dev_df,

    x_col='img_path',

    y_col='label',

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    # shuffle=True,

    class_mode='raw'

)

test_generator = test_datagen.flow_from_dataframe(

    test_df,

    x_col='img_path',

    y_col='label',

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    # shuffle=True,

    class_mode='raw'

)
def get_VGG16_model(input_shape):

    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    x = Flatten()(model.output)

    Dropout(0.5)(x)

    Dense(4096, activation='relu')(x)

    Dense(4096, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(model.input, output)

    

    return model





model = get_VGG16_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
class CustomCallback(Callback):

    def on_epoch_begin(self, epoch, logs=None):

        print()

        

    def on_epoch_end(self, epoch, logs=None):

        f1_score_result = 2 * (logs['precision'] * logs['recall']) / (logs['precision'] + logs['recall'])

        print(f"\nLoss: {logs['loss']}\nBinary Accuracy: {logs['binary_accuracy']}\nAUC: {logs['auc']}\nPrecision: {logs['precision']}\nRecall: {logs['recall']}\nF1 Score: {f1_score_result}")

        print()
model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

    loss='binary_crossentropy',

    metrics=['binary_accuracy', AUC(), Precision(), Recall()]

)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="loss",factor=0.1, patience=2, min_lr=0.00001, verbose=1

    ),

    CustomCallback(),

]





hist = model.fit(

    train_generator, 

    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

    batch_size=BATCH_SIZE, 

    validation_data=dev_generator, 

    epochs=30,

    callbacks=callbacks

)
# Evaluating labelled test data

test_results = model.evaluate(test_generator, steps=X_test.shape[0] // BATCH_SIZE)



print()



print(f'Test Loss: {test_results[0]}')

print(f'Test Binary Accuracy: {test_results[1]}')

print(f'Test ACU: {test_results[2]}')

print(f'Test Precision: {test_results[3]}')

print(f'Test Recall: {test_results[4]}')



f1_score_result = 2 * (test_results[3] * test_results[4]) / (test_results[3] + test_results[4])

print(f'Test F1 Score: {f1_score_result}')
!ls ./dogs-vs-cats/test
TEST_DIR = './dogs-vs-cats/test/test/'

test_imgs = [TEST_DIR + i for i in os.listdir(TEST_DIR)]
# Getting test data ready for prediction



x_test, y_test = process_images(test_imgs)



print(f'x_test length: {len(x_test)}, y_test length: {len(y_test)}')



X = np.asarray(x_test)



del x_test

gc.collect()



test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow(X, batch_size=BATCH_SIZE, shuffle=False)
predictions = model.predict(test_generator, verbose=1)

predictions = predictions.flatten()



results = []

for i in predictions:

    if i >= 0.5:

        results.append(1)

    else:

        results.append(0)
print(predictions[:10])

print()

print(results[:10])
# Cat = 0, Dog = 1



plt.figure(figsize=(15, 15))

for i, _img_path in enumerate(test_imgs[:25]):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.xlabel(f'Prediction: {results[i]}')

    # plt.xlabel(f'Prediction: {predictions[i]}')

    img = load_img(_img_path)

    plt.imshow(img)
submission = pd.read_csv('../input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')

submission.head()
for i in submission.index:

    # submission['label'].iloc[i] = results[i]

    submission['label'].iloc[i] = predictions[i]
submission.to_csv("sample_submission.csv", index=False)
model.save('model')       # SavedModel format

model.save('model.h5')    # HDF5 format