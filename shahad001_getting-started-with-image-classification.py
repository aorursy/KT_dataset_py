import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import zipfile  # for processing the zip files

import cv2  # for image processing

import matplotlib.pyplot as plt
%matplotlib inline

import random
import gc  # garbage collector
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
TRAIN_ZIP = '../input/dogs-vs-cats-redux-kernels-edition/train.zip'
zip_ref = zipfile.ZipFile(TRAIN_ZIP, 'r')
zip_ref.extractall('training')
zip_ref.close()
TEST_ZIP = '../input/dogs-vs-cats-redux-kernels-edition/test.zip'
zip_ref = zipfile.ZipFile(TEST_ZIP, 'r')
zip_ref.extractall('test_all')
zip_ref.close()
TRAIN_DIR = '/kaggle/working/training/train/'
TEST_DIR = '/kaggle/working/test_all/test/'

train_dogs = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_imgs = [TEST_DIR + i for i in os.listdir(TEST_DIR)]

# take trianing images from both classes
train_imgs = train_dogs[:2500] + train_cats[:2500]
random.shuffle(train_imgs)  # randomly shuffle the images

del train_dogs
del train_cats
gc.collect()
# view some sample images
import matplotlib.image as mpimg

for image in train_imgs[:10]:
    img = mpimg.imread(image)
    plt.figure()
    plt.imshow(img)
def process_images(list_of_images):
    x = []  # holds images
    y = []  # hold labels
    
    for image in list_of_images:
        x.append(
            cv2.resize(
                cv2.imread(image, cv2.IMREAD_COLOR),
                (height, width),
                interpolation=cv2.INTER_CUBIC
            )
        )
        
        if 'dog' in image:
            y.append(1)
        if 'cat' in image:
            y.append(0)
    
    return x, y
height = 200
width = 200
planes = 3  # for color image

X, y = process_images(train_imgs)

del train_imgs
gc.collect()
X = np.asarray(X)
y = np.asarray(y)

import seaborn as sns
sns.countplot(y)
plt.title('Labels')

print("Train image array shape:", X.shape)
print("Label array shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=2)

del X
del y
gc.collect()

num_of_train_imgs = len(X_train)
num_of_val_imgs = len(X_val)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])
model.summary()
# image generation

train_image_gen = ImageDataGenerator(rescale=1.0/255,
                                    rotation_range=50,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True)

val_image_gen= ImageDataGenerator(rescale=1.0/255)
batch_size = 32

train_image_generator = train_image_gen.flow(X_train, Y_train, batch_size=batch_size)
val_image_generator = val_image_gen.flow(X_val, Y_val, batch_size=batch_size)
history = model.fit_generator(train_image_generator,
                     steps_per_epoch=num_of_train_imgs // batch_size,
                     epochs=50,
                     validation_data=val_image_generator,
                     validation_steps=num_of_val_imgs // batch_size)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(train_acc) + 1)

# train and val acc
plt.plot(epochs, train_acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Val acc')
plt.title('Training and Validation accuracy')
plt.legend()

# train and val loss
plt.figure()
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Val loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
del X_train
del Y_train
del X_val
del Y_val

gc.collect()
# print(test_imgs[0][30:-4])
ids = [a[30:-4] for a in test_imgs]
print(ids[:10])
x_test, y_test = process_images(test_imgs)
X = np.asarray(x_test)

del x_test
gc.collect()

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow(X, batch_size=1, shuffle=False)
predictions = model.predict_generator(test_generator, verbose=1)
predictions = predictions.flatten()
res = []
for i in predictions:
    if i >=0.5:
        res.append(1)
    else:
        res.append(0)
sub_df = pd.DataFrame({"id": ids, "label": predictions})
sub_df.to_csv("dogs_cats_predict.csv", index = False)
sub_df.head()