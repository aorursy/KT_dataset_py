# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random
import tensorflow as tf

from sklearn.model_selection import train_test_split                       # used to split dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img        # used to feed jpg images into model

from tensorflow.keras.applications import Xception                         # load pretrained model

from matplotlib import pyplot as plt                                       # for data visualization



from skimage.transform import rotate, AffineTransform, warp                # for data augmentation

from skimage.transform import resize                                       # resize image

from skimage import io                                                     # for saving images



from tensorflow.keras.models import Sequential

from keras.optimizers import SGD

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

train
train_dir='/kaggle/input/jpeg-melanoma-512x512/train/'

test_dir='/kaggle/input/jpeg-melanoma-512x512/test/'
#make a dataframe that has image location

images_df = pd.DataFrame()

images_df['image_address'] = train_dir + train['image_name'] + '.jpg'

images_df['target'] = train['target']

images_df
print('Number of malignant cases = ', images_df['target'].sum())

print('Number of benign cases    = ', images_df['target'].count() - images_df['target'].sum())

print('Ratio of malignant cases   = ', images_df['target'].sum()/images_df['target'].count())
# let's use only ~1200 benign images



#lets get a mask with all the benign cases as true

benign_cases_truth = (images_df['target'] == 0).iloc[:2000]

print('we use only', benign_cases_truth.sum(), 'benign cases')



#apply the mask to get ~2000 benign cases

benign_cases = (images_df[:].iloc[:1200])[:][benign_cases_truth]



# use all the malignant cases:

malignant_cases = images_df[:][images_df['target'] == 1]

print('we use', len(malignant_cases.index), 'malignant cases')



images_df = benign_cases.copy()

images_df = images_df.append(malignant_cases)

# don't worry about the order, train_test_split shuffles by default

images_df
# split the data into training and dev set so we can validate our model

X_train, X_dev, y_train, y_dev = train_test_split(images_df, images_df['target'], 

                                                  test_size=0.2, random_state=42)
input_shape = (512, 512)

input_shape_with_channels = (512, 512, 3)
os.makedirs('./train/benign')

os.makedirs('./train/melanoma')

os.makedirs('./test/benign')

os.makedirs('./test/melanoma')
def augment_and_save(path, label, train_or_test):

    

    if label == 0:

        label = 'benign'

    elif label == 1:

        label = 'melanoma'

    image_name = path[-16:-4]

    save_location = train_or_test+'/'+label+'/'+image_name

    

    #read the image

    image_string = tf.io.read_file(path)

    image = tf.image.decode_jpeg(image_string,channels=3)

    

    #brightness

    brightened = tf.image.random_brightness(image, 0.5)



    #flipped

    flipLR = tf.image.flip_left_right(image)

    

    #saturation

    saturated = tf.image.adjust_saturation(image, random.randint(2,5))

    

    #central crop

    cropped = tf.image.central_crop(image, central_fraction=0.4)

    

    #updown

    updown = tf.image.flip_up_down(image)



    #save image

    save_img(save_location+'_brightened.jpg', brightened)

    save_img(save_location+'_flipLR.jpg',  flipLR)    

    save_img(save_location+'_saturated.jpg',  saturated)

    save_img(save_location+'_cropped.jpg', cropped)

    save_img(save_location+'_updown.jpg', updown)
# to make training data

X_train.apply(lambda row : augment_and_save(row['image_address'], 

                                  row['target'], 'train'), axis = 1)



# similarly do the same for dev data

X_dev.apply(lambda row : augment_and_save(row['image_address'], 

                                  row['target'], 'test'), axis = 1)
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)



train_it = datagen.flow_from_directory('/kaggle/working/train/',

                                       class_mode='binary', batch_size=64, target_size=(512, 512),

                                       subset='training')

valid_it = datagen.flow_from_directory('/kaggle/working/train/',

                                       class_mode='binary', batch_size=64, target_size=(512, 512),

                                       subset='validation')

test_it = datagen.flow_from_directory('/kaggle/working/test/', 

                                      class_mode='binary', batch_size=64, target_size=(512, 512))
model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# compile model

# opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_it, validation_data=valid_it, epochs=50, verbose=0)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(50)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)

print('> %.3f' % (acc * 100.0))
model.save('final_model.h5')
model_path = '/kaggle/working/final_model.h5'
# from tensorflow.keras.preprocessing import image

# list_images = sorted(os.listdir(test_dir))

# test_images = []

# for img in list_images:

#     img = os.path.join(test_dir, img)

#     img = image.load_img(img, target_size=(256, 256))

#     img = image.img_to_array(img)

#     img = np.expand_dims(img, axis=0)

#     test_images.append(img)



# images = np.vstack(test_images)
# preds = model.predict(images)

# print(preds)