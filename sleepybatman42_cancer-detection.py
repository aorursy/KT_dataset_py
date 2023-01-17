import os

import shutil

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import math



from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator



import keras

from keras.models import Sequential, load_model

from keras.models import *

from keras.layers import *



import utilities
test_df = pd.read_csv("../input/histopathologic-cancer-detection/sample_submission.csv", dtype=str)

labeled_df = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv", dtype=str)
test_df.head()
labeled_df.head()
labeled_df.id = labeled_df.id + '.tif'

test_id = test_df.id

test_df.id = test_df.id + '.tif'

print(labeled_df.head())

print(test_df.head())
train_path = "../input/histopathologic-cancer-detection/train/"

test_path = "../input/histopathologic-cancer-detection/test/"
print(len(os.listdir(train_path)))

print(len(os.listdir(test_path)))
labeled_df.label.value_counts() / len(labeled_df.label)
labeled_df.label.value_counts()
n = 80000

negative_sample = labeled_df.loc[labeled_df.label == '0', :].sample(n, random_state=1)

positive_sample = labeled_df.loc[labeled_df.label == '1', :].sample(n, random_state=1)

labeled_sample = pd.concat([negative_sample, positive_sample], axis=0).reset_index(drop=True)



train_df, valid_df = train_test_split(labeled_sample, test_size=0.2, random_state=1, stratify=labeled_sample.label)



print(train_df.shape)

print(valid_df.shape)
base_dir = 'images/'

train_dir = 'images/train/'

valid_dir = 'images/valid/'

test_dir = 'images/test/'



os.mkdir(base_dir)

os.mkdir(train_dir)

os.mkdir(valid_dir)

os.mkdir(test_dir)



os.mkdir(train_dir + 'negative')

os.mkdir(train_dir + 'positive')

os.mkdir(valid_dir + 'negative')

os.mkdir(valid_dir + 'positive')

os.mkdir(test_dir + 'unlabeled')
%%time 



for i in range(len(train_df.id)):

    

    src = train_path + train_df.id.iloc[i]

        

    if train_df.label.iloc[i] == '0':    

        dest = train_dir + 'negative/' + train_df.id.iloc[i]

    else: 

        dest = train_dir + 'positive/' + train_df.id.iloc[i]

        

    shutil.copyfile(src, dest)



print(len(os.listdir(train_dir + 'negative')))

print(len(os.listdir(train_dir + 'positive')))
%%time 



for i in range(len(valid_df.id)):

    

    src = train_path + valid_df.id.iloc[i]

        

    if valid_df.label.iloc[i] == '0':    

        dest = valid_dir + 'negative/' + valid_df.id.iloc[i]

    else: 

        dest = valid_dir + 'positive/' + valid_df.id.iloc[i]

        

    shutil.copyfile(src, dest)



print(len(os.listdir(valid_dir + 'negative')))

print(len(os.listdir(valid_dir + 'positive')))
bs = 64



train_datagen = ImageDataGenerator(rescale=1/255)

valid_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)



train_generator = train_datagen.flow_from_directory(

    directory = train_dir,

    batch_size = bs,

    shuffle = True,

    class_mode = "binary",

    target_size = (96,96))



valid_generator = train_datagen.flow_from_directory(

    directory = valid_dir,

    batch_size = bs,

    shuffle = True,

    class_mode = "binary",

    target_size = (96,96))
tr_size = 128000 

va_size = 32000



tr_steps = math.ceil(tr_size / bs)

va_steps = math.ceil(va_size / bs)





print('Number of training batches:  ', tr_steps)

print('Number of validation batches:', va_steps)

def training_images(seed):

    np.random.seed(seed)

    train_generator.reset()

    imgs, labels = next(train_generator)

        

    plt.figure(figsize=(12,12))

    for i in range(16):

        plt.subplot(4,4,i+1)

        plt.imshow(imgs[i,:,:,:])

        if(labels[i] == 1):

            plt.text(0, -5, 'Positive', color='r')

        else:

            plt.text(0, -5, 'Negative', color='b')

        plt.axis('off')

    plt.show()



training_images(1)
np.random.seed(1)



cnn = Sequential()

cnn.add(Cropping2D(cropping=((32,32), (32,32)), input_shape=(96,96,3)))

cnn.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

cnn.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

cnn.add(MaxPooling2D(2,2))

#cnn.add(Dropout(0.25))

cnn.add(BatchNormalization())



cnn.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

cnn.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

cnn.add(MaxPooling2D(2,2))

#cnn.add(Dropout(0.25))

cnn.add(BatchNormalization())



cnn.add(Flatten())

cnn.add(Dense(512, activation='relu'))

#cnn.add(Dropout(0.25))

cnn.add(BatchNormalization())

cnn.add(Dense(1, activation='sigmoid'))



cnn.summary()
%%time 



opt = keras.optimizers.Adam(0.002)

cnn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



h1 = cnn.fit_generator(train_generator, steps_per_epoch=tr_steps, epochs=5,

                       validation_data=valid_generator, validation_steps=va_steps, 

                       verbose=1)
utilities.vis_training([h1])
opt = keras.optimizers.Adam(lr=0.0002)



h2 = cnn.fit_generator(train_generator, steps_per_epoch=tr_steps, epochs=20,

                       validation_data=valid_generator, validation_steps=va_steps, 

                       verbose=1)
utilities.vis_training([h1,h2])
opt = keras.optimizers.Adam(lr=0.00001)



h3 = cnn.fit_generator(train_generator, steps_per_epoch=tr_steps, epochs=15,

                       validation_data=valid_generator, validation_steps=va_steps, 

                       verbose=1)
utilities.vis_training([h1,h2, h3])
cnn.save('cnn_v01.h5')
shutil.rmtree('/kaggle/working/images/train')

shutil.rmtree('/kaggle/working/images/valid')
%%time



for i in range(len(test_df.id)):

    

    src = test_path + test_df.id.iloc[i]

    dest = test_dir + 'unlabeled/' + test_df.id.iloc[i]

    shutil.copyfile(src, dest)

    

print(len(os.listdir(test_dir + 'unlabeled')))
bs = 64



test_datagen = ImageDataGenerator(rescale=1/255)



test_generator = test_datagen.flow_from_directory(

    directory = test_dir,

    batch_size = bs,

    shuffle = False,

    class_mode = None,

    target_size = (96,96))
te_size = 57458

te_steps = math.ceil(te_size / bs)

print('Number of test batches:    ', te_steps)
test_pred = cnn.predict_generator(test_generator, steps = te_steps, verbose=1)
test_pred[:5]
test_fnames = test_generator.filenames

test_fnames[:5]
test_fnames = [x.split('.')[0] for x in test_fnames]

test_fnames = [x.split('/')[1] for x in test_fnames]

test_fnames[:5]
print(test_pred.shape)

#pred_classes = np.argmax(test_pred, axis=1)

pred_classes= np.where(test_pred > 0.5, 1, 0)



print(pred_classes[:5])



print(np.sum(pred_classes == 0))

print(np.sum(pred_classes == 1))
print(len(test_fnames))

print(pred_classes.shape)
submission = pd.DataFrame({

    'id':test_fnames,

    'label':pred_classes.reshape(-1,)

})

submission.head()
submission.to_csv('submission.csv', index=False)
shutil.rmtree('/kaggle/working/images')