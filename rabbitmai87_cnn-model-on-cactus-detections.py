import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from PIL import Image

import matplotlib.pyplot as plt

from glob import glob



from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import DenseNet169
base_dir = os.path.join('..', 'input')
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))

train_df.head()
#Let's see how many training data:

print("There are {} training images".format(len(train_df)))
train_images = os.path.join(base_dir, 'train/train')

test_images = os.path.join(base_dir, 'test/test')



print("There are {} images in the train folder.".format(len(os.listdir(train_images))))

print("There are {} images in the test folder.".format(len(os.listdir(test_images))))
#create a column in dataframe to store image paths

#let's test the glob function first:

i = 0

for x in glob(os.path.join(train_images,'*.jpg')):

    print(x)

    i += 1

    

    if i == 5:

        break
image_id_path = {os.path.basename(x): x for x in glob(os.path.join(train_images, '*.jpg'))}
train_df['path'] = train_df['id'].map(image_id_path.get)
train_df.head()
train_df['images'] = train_df['path'].map(lambda x: np.array(Image.open(x).resize((32,32))))
train_df['images'] = train_df['images'] / 255
#each image's shape

train_df['images'][0].shape
train_df.isnull().sum()
train_df['has_cactus'].value_counts().plot(kind='bar')
n_rows = 5

n_cols = 5



fig, ax = plt.subplots(n_rows, n_cols, figsize=(15,16))



i = 0

for each_row in ax:

    

    for each_row_col in each_row:

        each_row_col.imshow(train_df['images'][i])

        

        if train_df['has_cactus'][i] == 1:

            each_row_col.set_title('Cactus')

        else:

            each_row_col.set_title('None')

        

        each_row_col.axis('off')

        i += 1
test_df = pd.read_csv(os.path.join(base_dir,'sample_submission.csv'))

test_df.head()
test_imageid_path = {os.path.basename(x):x for x in glob(os.path.join(test_images,'*.jpg'))}
test_df['path'] = test_df['id'].map(test_imageid_path.get)
test_df['images'] = test_df['path'].map(lambda x: np.array(Image.open(x)))
test_df.head()
#sample one test image

print("test image's shape: ", test_df['images'][0].shape)
plt.imshow(test_df['images'][0])
#check if there is null values in test dataframe:

test_df.isnull().sum()
#normalize test image pixels

test_df['images'] = test_df['images'] / 255
test_images = np.asarray(test_df['images'].tolist())
test_images.shape
features = train_df['images']

labels = train_df['has_cactus']
features = np.asarray(features.tolist())
labels = np.asarray(labels.tolist())
x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.1, random_state=1234)
print("x_train size: ", len(x_train))

print("x_valid size: ", len(x_valid))
x_train.shape
callbacks = [ReduceLROnPlateau(monitor= 'val_acc',

                              patience=3,

                              verbose=1,

                              factor=0.5)

#              ,EarlyStopping(monitor='val_acc',

#                            patience=20,

#                           verbose=1,

#                           mode='auto',

#                           restore_best_weights=True)

            ]
# datagen = ImageDataGenerator(featurewise_center=False,

#                              samplewise_center=False,

#                              featurewise_std_normalization=False,

#                              samplewise_std_normalization=False,

#                              rotation_range=40,

#                              zoom_range=0.3,

#                              width_shift_range=0.3,

#                              height_shift_range=0.3,

#                              vertical_flip=True

#                              #horizontal_flip=True

#                             )
# datagen.fit(x_train)
def visual(model_history):

    

    fig, ax = plt.subplots(1,2, figsize=(15,6))

    

    #model accuracy

    acc = model_history.history['acc']

    valid_acc = model_history.history['val_acc']

    

    ax[0].plot(range(1, len(model_history.history['acc'])+1), acc, label='train accuracy')

    ax[0].plot(range(1, len(model_history.history['acc'])+1), valid_acc, label='validation accuracy')

    ax[0].set_title('Model Accuracy')

    ax[0].set_xlabel('epochs')

    ax[0].set_ylabel('accuracy')

    ax[0].legend()

    

    #model loss

    loss = model_history.history['loss']

    valid_loss = model_history.history['val_loss']

    

    ax[1].plot(range(1, len(model_history.history['acc'])+1), loss, label='train loss')

    ax[1].plot(range(1, len(model_history.history['acc'])+1), valid_loss, label='validation loss')

    ax[1].set_title('Model Loss')

    ax[1].set_xlabel('epochs')

    ax[1].set_ylabel('loss')

    ax[1].legend()
#for cnn model, we are going to use x_train, x_valid, y_train, y_valid

print("x_train's shape: ", x_train.shape)

print("y_train's shape: ", y_train.shape)

print("\nx_valid's shape: ", x_valid.shape)

print("y_valid's shape: ", y_valid.shape)
input_shape = x_train.shape[1:]



cnn = Sequential([Conv2D(32, kernel_size=(2,2), activation='relu', padding='same', input_shape = input_shape),

                  Conv2D(32, kernel_size=(2,2), activation='relu', padding='same'),

                  MaxPool2D(pool_size=(2,2)),

                  Dropout(0.2),

                  

                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),

                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),

                  MaxPool2D(pool_size=(2,2)),

                  Dropout(0.2),

                  

                  Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'),

                  Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),

                  MaxPool2D(pool_size=(2,2)),

                  Dropout(0.5),

                  

#                   Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),

#                   Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),

#                   MaxPool2D(pool_size=(2,2)),

#                   Dropout(0.5),

                  

                  Flatten(),

                  Dense(64, activation='relu'),

                  Dropout(0.5),

                  Dense(28, activation='relu'),

                  Dropout(0.5),

                  Dense(1, activation='sigmoid')

                 ])
cnn.summary()
#compile the cnn model

cnn.compile(optimizer=Adam(lr=0.001), loss = "binary_crossentropy", metrics=['acc'])
#fit the cnn model

cnn_history = cnn.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_valid,y_valid), callbacks=callbacks)
# #fit the cnn model with data augmentations

# cnn_history = cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=64), epochs=50, validation_data=(x_valid, y_valid),

#                                 steps_per_epoch = x_train.shape[0] // 64,

#                                 callbacks = callbacks)
visual(cnn_history)
test_pred = cnn.predict_classes(test_images)
test_df['has_cactus'] = np.squeeze(test_pred)
test_df.head()
submission_df = test_df[['id','has_cactus']]
submission_df.to_csv('cactus_detections.csv', index=False)