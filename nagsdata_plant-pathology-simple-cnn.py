# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

import matplotlib.pyplot as plt
train_csv = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")

train_csv.head()
test_csv = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")

test_csv.head()
image = plt.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg')

plt.imshow(image)

print(image.shape)
def image_resize(img, size = (None, None), ratio=3):

    if size[0] is None:

        resize_ratio = ratio

        resize_height = int(img.shape[0]/resize_ratio)

        resize_width = int(img.shape[1]/resize_ratio)

        print(f"height: {resize_height}, width: {resize_width}")

    else:

        resize_height = size[0]

        resize_width = size[1]



    img_resize = tf.image.resize(img, [resize_height,resize_width]).numpy()

    img_resize = img_resize.astype(np.uint8)

    return(img_resize)
plt.figure(1, figsize=(10,10))

plt.subplot(221)

plt.imshow(image_resize(image, ratio = 3))



plt.subplot(222)

plt.imshow(image_resize(image, ratio = 4))

plt.show()



plt.subplot(223)

plt.imshow(image_resize(image, ratio = 5))



plt.subplot(224)

plt.imshow(image_resize(image, ratio = 6))

plt.show()
img_height = 227

img_width = 341

plt.imshow(image_resize(image, size=(img_height, img_width)))
train_resized = []



for img_id in train_csv['image_id'].to_list():

    image = plt.imread(f'/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg')

    train_resized.append(image_resize(image, (img_height, img_width)))



print(len(train_resized))



test_resized = []



for img_id in test_csv['image_id'].to_list():

    image = plt.imread(f'/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg')

    test_resized.append(image_resize(image, (img_height, img_width)))



print(len(test_resized))
x_train = np.ndarray(shape = (len(train_resized), img_height, img_width, 3), dtype=np.float32)

x_test = np.ndarray(shape = (len(test_resized), img_height, img_width, 3), dtype=np.float32)



for i in range(len(train_resized)):

    x_train[i] = img_to_array(train_resized[i])



for i in range(len(test_resized)):

    x_test[i] = img_to_array(test_resized[i])



x_train = x_train/255

x_test = x_test/255



print(x_train.shape)

print(x_test.shape)
y_train = train_csv.iloc[:,1:]

y_train.head()
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

from tensorflow.keras.applications import InceptionResNetV2



resnet = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')



model = Sequential()

model.add(resnet)

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(y_train.shape[1], activation='softmax'))



model.layers[0].trainable = False



model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics='accuracy')
from sklearn.model_selection import train_test_split



train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size = 0.2)



print(train_x.shape)

print(train_y.shape)



print(val_x.shape)

print(val_y.shape)

datagen = ImageDataGenerator(rotation_range=25,

                             shear_range=.20,

                             zoom_range=.20,

                             width_shift_range=.20,

                             height_shift_range=.20,

                             horizontal_flip=True,

                             vertical_flip=True

                            )



#train_datagen = datagen.flow(rain, y_train, batch_size=42, seed=42)



batch_size = 24

datagen_without_aug = ImageDataGenerator()



train_datagen = datagen_without_aug.flow(train_x, train_y, batch_size=batch_size)



val_datagen = datagen_without_aug.flow(val_x, val_y, batch_size=batch_size)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit_generator(train_datagen, 

                              epochs=6,

                              steps_per_epoch=train_x.shape[0]//batch_size,

                              validation_data = val_datagen,

                              validation_steps = val_x.shape[0]//batch_size,

                              callbacks = [callback]

                   )
history_df = pd.DataFrame(history.history)

history_df.head()
plt.plot(history_df.index, history_df['accuracy'])

plt.plot(history_df.index, history_df['val_accuracy'])

plt.show()
plt.plot(history_df.index, history_df['loss'])

plt.plot(history_df.index, history_df['val_loss'])

plt.show()
y_preds = model.predict(x_test)
y_preds
res = pd.DataFrame()

res['image_id'] = test_csv['image_id']

res['healthy'] = y_preds[:, 0]

res['multiple_diseases'] = y_preds[:, 1]

res['rust'] = y_preds[:, 2]

res['scab'] = y_preds[:, 3]

res.to_csv('submission.csv', index=False)