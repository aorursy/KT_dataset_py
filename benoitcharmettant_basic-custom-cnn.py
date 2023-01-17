# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import shutil

from tqdm import tqdm

from PIL import Image



DATA_PATH = "../input/"

TRAIN_PATH = DATA_PATH + 'train_images/'



print(os.listdir(DATA_PATH))



from glob import glob 

from skimage.io import imread

import gc



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
base_tile_dir = DATA_PATH + 'train_images/'



df = pd.DataFrame({'path': glob(base_tile_dir +'/*.png')})



df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])



labels = pd.read_csv(DATA_PATH + "train.csv")

labels = labels.rename(index=str, columns={'id_code':'id', 'diagnosis':'label'})



df_data = df.merge(labels, on = "id")



df_data.head()
SAMPLE_SIZE = 150 # load 80k negative examples



# take a random sample of class 0 with size equal to num samples in class 1

df_0 = df_data[df_data['label'] == 0].sample(1805, random_state = 101)

# filter out class 1

df_1 = df_data[df_data['label'] == 1].sample(370, random_state = 101)

# filter out class 2

df_2 = df_data[df_data['label'] == 2].sample(999, random_state = 101)

# filter out class 3

df_3 = df_data[df_data['label'] == 3].sample(193, random_state = 101)

# filter out class 4

df_4 = df_data[df_data['label'] == 4].sample(295, random_state = 101)



# concat the dataframes

df_data = shuffle(pd.concat([df_0, df_1, df_2, df_3, df_4], axis=0).reset_index(drop=True))



print(df_data.head())

print(df_data.label.value_counts())



# train_test_split # stratify=y creates a balanced validation set.

y = df_data['label']

df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)



# Create directories

train_path = 'base_dir/train'

valid_path = 'base_dir/valid'

test_path = '../input/test'

for fold in [train_path, valid_path]:

    for subf in ["0", "1","2","3","4"]:

        os.makedirs(os.path.join(fold, subf))
# Set the id as the index in df_data

df_data.set_index('id', inplace=True)

df_data.head()


IMAGE_SIZE = 192



for image in tqdm(df_train['id'].values):

    # the id in the csv file does not have the .tif extension therefore we add it here

    fname = image + '.png'

    label = str(df_data.loc[image,'label']) # get the label for a certain image

    src = os.path.join(TRAIN_PATH, fname)

    dst = os.path.join(train_path, label, fname)

    

    pil_im = Image.open(src)

    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))

    resized_image.save(dst)



for image in tqdm(df_val['id'].values):

    fname = image + '.png'

    label = str(df_data.loc[image,'label']) # get the label for a certain image

    src = os.path.join(TRAIN_PATH, fname)

    dst = os.path.join(valid_path, label, fname)

    

    pil_im = Image.open(src)

    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))

    resized_image.save(dst)

from keras.preprocessing.image import ImageDataGenerator



num_train_samples = len(df_train)

num_val_samples = len(df_val)

train_batch_size = 32

val_batch_size = 32



train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)



datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,

                            horizontal_flip=True,

                            vertical_flip=True)



train_gen = datagen.flow_from_directory(train_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=train_batch_size,

                                        class_mode='categorical')



val_gen = datagen.flow_from_directory(valid_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=val_batch_size,

                                        class_mode='categorical')



# Note: shuffle=False causes the test dataset to not be shuffled

test_gen = datagen.flow_from_directory(valid_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=1,

                                        class_mode='categorical',

                                        shuffle=False)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.layers import Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam



kernel_size = (3,3)

pool_size= (2,2)

ini_filters = 32

first_filters = 32

second_filters = 64

third_filters = 128



dropout_conv = 0.3

dropout_dense = 0.5



model = Sequential()



model.add(Conv2D(ini_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))

model.add(Conv2D(ini_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))

model.add(Conv2D(first_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(MaxPool2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



#model.add(GlobalAveragePooling2D())

model.add(Flatten())

model.add(Dense(256, use_bias=False))

model.add(BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(dropout_dense))

model.add(Dense(5, activation = "sigmoid"))



# Compile the model

model.compile(Adam(0.01), loss = "categorical_crossentropy", metrics=["accuracy"])



print("Done !")
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)

reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 

                    validation_data=val_gen,

                    validation_steps=val_steps,

                    epochs=30,

                   callbacks=[reducel, earlystopper])
from sklearn.metrics import roc_curve, auc, roc_auc_score

import matplotlib.pyplot as plt



# make a prediction

y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)



print(y_pred_keras)
base_test_dir = '../input/test_images/'



test_files = glob(os.path.join(base_test_dir,'*.png'))



os.makedirs('valid/')



for image in tqdm(test_files):

    fname = image.split('/')[-1]

    

    src = os.path.join(base_test_dir, fname)

    dst = os.path.join("valid/",fname)

    

    pil_im = Image.open(src)

    resized_image = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE))

    resized_image.save(dst)

    

test_files = glob(os.path.join('valid','*.png'))





submission = pd.DataFrame()

file_batch = 20

max_idx = len(test_files)

for idx in range(0, max_idx, file_batch):

    print("Indexes: %i - %i"%(idx, idx+file_batch))

    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]})

    test_df['id_code'] = test_df.path.map(lambda x: x.split('/')[1].split(".")[0])

    test_df['image'] = test_df['path'].map(imread)

    K_test = np.stack(test_df["image"].values)

    K_test = (K_test - K_test.mean()) / K_test.std()

    predictions = model.predict(K_test)

    

    pred = []

    

    for l in predictions:

        pred.append(np.argmax(l))

    

    

    test_df['diagnosis'] = pred

    submission = pd.concat([submission, test_df[["id_code", "diagnosis"]]])

submission.head()
#submission

# Delete the test_dir directory we created to prevent a Kaggle error.

# Kaggle allows a max of 500 files to be saved.

submission.to_csv("submission.csv", index = False, header = True)
df = pd.read_csv("submission.csv")

print(df["diagnosis"].value_counts())



print(predictions)

print(pred)
shutil.rmtree(train_path)

shutil.rmtree(valid_path)

shutil.rmtree('valid/')