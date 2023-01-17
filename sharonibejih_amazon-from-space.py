import numpy as np 

import pandas as pd 

import os

import sys

from tqdm import tqdm

import pathlib



import matplotlib.pyplot as plt

%matplotlib inline

import cv2



from keras.applications.vgg16 import VGG16

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



from sklearn.utils import shuffle

from sklearn.metrics import fbeta_score
for dirname, _, filenames in os.walk('/kaggle/input/planets-dataset'):

    for filename in filenames:

        os.path.join(dirname, filename)
!ls /kaggle/input/planets-dataset
train_df = pd.read_csv("/kaggle/input/planets-dataset/planet/planet/train_classes.csv")

test_df = pd.read_csv("/kaggle/input/planets-dataset/planet/planet/sample_submission.csv")
train_classes = train_df[:]['tags']



no_classes = len(train_classes.unique())

print(f'Given {len(train_classes)} samples, there are {no_classes} unique classes.', '\n')



train_df.head()
# Split the tags column to get the unique labels

flatten = lambda l: [item for sublist in l for item in sublist]

labels = list(set(flatten([l.split(' ') for l in train_df['tags'].values])))



# Mapping the label value counts

label_map = {l: i for i, l in enumerate(labels)}

print(f'labels = {labels},\n length = {len(labels)}', '\n')



print(f'label_map = {label_map},\n length = {len(label_map)}')
keys = list(label_map.keys())

values = list(label_map.values())

labels_df = pd.DataFrame({'labels':keys, 'freq':values})

labels_df = labels_df.sort_values('freq')



plt.rcParams['figure.figsize']=(14,5)

plt.xticks(rotation=90)

plt.bar('labels', 'freq', data=labels_df)
new_style = {'grid': False}

plt.rc('axes', **new_style)

_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(15, 15))

i = 0

for f, tags in train_df[:9].values:

    img = cv2.imread('/kaggle/input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(f))

    ax[i // 3, i % 3].imshow(img)

    ax[i // 3, i % 3].set_title('{} - {}'.format(f, tags))

  

    i += 1

    

plt.show()
# Load the train-jpg file path



train_img_dir = pathlib.Path('/kaggle/input/planets-dataset/planet/planet/train-jpg')



train_img_path = sorted(list(train_img_dir.glob('*.jpg')))



train_img_count = len(train_img_path)

print(train_img_count)
# first test jpg file path

test_img_dir = pathlib.Path('/kaggle/input/planets-dataset/planet/planet/test-jpg')



test_img_path = sorted(list(test_img_dir.glob('*.jpg')))



test_img_count = len(test_img_path)

print(test_img_count)
# second test jpg file path



test_add_img_dir = pathlib.Path('/kaggle/input/planets-dataset/test-jpg-additional')



test_add_img_path = sorted(list(test_add_img_dir.glob('*/*.jpg')))



test_add_img_count = len(test_add_img_path)

print(test_add_img_count)
# Ensure the number of jpg images are equal to the number of samples in the csv file for each data set



# train

assert len(train_img_path) == len(train_df)



# test

assert len(test_img_path)+len(test_add_img_path) == len(test_df)
input_size = 64

input_channels = 3



batch_size = 64
x_train = []

y_train = []



for f, tags in tqdm(train_df.values, miniters=1000):

    img = cv2.imread('/kaggle/input/planets-dataset/planet/planet/train-jpg/{}.jpg'.format(f))

    img = cv2.resize(img, (input_size, input_size))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1

    x_train.append(img)

    y_train.append(targets)

        

x_train = np.array(x_train, np.float32)

y_train = np.array(y_train, np.uint8)



print(x_train.shape)

print(y_train.shape)
x_test = []



test_jpg_dir = '/kaggle/input/planets-dataset/planet/planet/test-jpg'

test_image_names = os.listdir(test_jpg_dir)



n_test = len(test_image_names)

test_classes = test_df.iloc[:n_test, :]

add_classes = test_df.iloc[n_test:, :]



test_jpg_add_dir = '/kaggle/input/planets-dataset/test-jpg-additional/test-jpg-additional'

test_add_image_names = os.listdir(test_jpg_add_dir)



for img_name, _ in tqdm(test_classes.values, miniters=1000):

    img = cv2.imread(test_jpg_dir + '/{}.jpg'.format(img_name))

    x_test.append(cv2.resize(img, (64, 64)))

    

for img_name, _ in tqdm(add_classes.values, miniters=1000):

    img = cv2.imread(test_jpg_add_dir + '/{}.jpg'.format(img_name))

    x_test.append(cv2.resize(img, (64, 64)))



x_test = np.array(x_test, np.float32)

print(x_test.shape)
# split the train data into train and validation data sets

X_train = x_train[ :33000]

Y_train = y_train[ :33000]



X_valid = x_train[33000: ]

Y_valid = y_train[33000: ]
base_model = VGG16(include_top=False,

                   weights='imagenet',

                   input_shape=(input_size, input_size, input_channels))



model = Sequential()

model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))



model.add(base_model)

model.add(Flatten())

model.add(Dropout(0.5))



model.add(Dense(17, activation='sigmoid'))
from keras.optimizers import SGD

opt  = SGD(lr=0.01)



model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])

    

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),

                ModelCheckpoint(filepath='weights/best_weights',

                                 save_best_only=True,

                                 save_weights_only=True)]

model.summary()
history = model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),

                  batch_size=batch_size,verbose=2, epochs=15,callbacks=callbacks,shuffle=True)
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
p_valid = model.predict(X_valid, batch_size = batch_size, verbose=1)



print(fbeta_score(Y_valid, np.array(p_valid) > 0.18, beta=2, average='samples'))
y_pred = []

p_test = model.predict(x_test, batch_size=batch_size, verbose=2)

y_pred.append(p_test)
result = np.array(y_pred[0])

for i in range(1, len(y_pred)):

    result += np.array(y_pred[i])

result = pd.DataFrame(result, columns=labels)
# Translating the probability predictions to the unique labels

preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):

    a = result.loc[[i]]

    a = a.apply(lambda x: x>0.2, axis=1)

    a = a.transpose()

    a = a.loc[a[i] == True]

    ' '.join(list(a.index))

    preds.append(' '.join(list(a.index)))
# Replacing the tags columns with the predicted labels

test_df['tags'] = preds

test_df.head()
# Converting the dataframe to a csv file for submission

test_df.to_csv('amazon_sample_submission.csv', index=False)