import glob

import os

import pandas as pd

from sklearn.utils import shuffle

import numpy as np

import tensorflow as tf

from sklearn.linear_model import LogisticRegression

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,GlobalMaxPooling2D

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras import utils

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.vgg16 import VGG16



import matplotlib.pyplot as plt

%matplotlib inline
for x in glob.glob('../input/intel-image-classification/seg_train/seg_train/*'):

    print (os.path.basename(x))
class_to_int = {'buildings' : 0, 'forest' : 1, 'glacier' : 2, 'mountain' : 3, 'sea' : 4, 'street' : 5 }

int_to_class = {0 : 'buildings', 1 : "forest", 2: "glacier", 3 : "mountain", 4 : "sea", 5 : "street"}
train_path = '../input/intel-image-classification/seg_train/seg_train/'

test_path = '../input/intel-image-classification/seg_test/seg_test/'

data_train_dict = {}

data_train_dict['cat'] = []

data_train_dict['cat_num'] = []

data_train_dict['file_path'] = []



for k in class_to_int.keys():

    data_train_dict['file_path'] += list(glob.glob(train_path + k + "/*"))

    data_train_dict['cat'] += list(np.repeat(k,len(list(glob.glob(train_path + k + "/*")))))

    data_train_dict['cat_num'] += list(np.repeat(class_to_int[k],len(list(glob.glob(train_path + k + "/*")))))

    
data_train = pd.DataFrame(data_train_dict)

data_train = shuffle(data_train, random_state = 44)

data_train.head()
data_test_dict = {}

data_test_dict['cat'] = []

data_test_dict['cat_num'] = []

data_test_dict['file_path'] = []



for k in class_to_int.keys():

    data_test_dict['file_path'] += list(glob.glob(test_path + k + "/*"))

    data_test_dict['cat'] += list(np.repeat(k,len(list(glob.glob(test_path + k + "/*")))))

    data_test_dict['cat_num'] += list(np.repeat(class_to_int[k],len(list(glob.glob(test_path + k + "/*")))))

    
data_test = pd.DataFrame(data_test_dict)

data_test = shuffle(data_test, random_state = 44)

data_test.head()
y_train = utils.to_categorical(data_train['cat_num'])

y_test = utils.to_categorical(data_test['cat_num'])
data_train['cat_num'][2]
y_train[2]
images_train = np.array([img_to_array(

                    load_img(img, target_size=(150,150))

                    ) for img in data_train['file_path'].values.tolist()])
images_test = np.array([img_to_array(

                    load_img(img, target_size=(150,150))

                    ) for img in data_test['file_path'].values.tolist()])
images_test.shape
images_train = images_train.astype('float32')/255.0

images_test = images_test.astype('float32')/255.0
plt.imshow(images_train[0]);

plt.grid(True);

plt.xticks([]);

plt.yticks([]);

plt.title("Class = " + int_to_class[np.argmax(y_train[0])]);
plt.imshow(images_train[1]);

plt.grid(True);

plt.xticks([]);

plt.yticks([]);

plt.title("Class = " + int_to_class[np.argmax(y_train[1])]);
total_classes = len(class_to_int.keys())
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3), padding = "same"))

model.add(Activation('relu')) # this is just different syntax for specifying the activation function

model.add(BatchNormalization())

model.add(Dropout(0.6))



model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 3), padding = "same"))

model.add(Activation('relu')) # this is just different syntax for specifying the activation function

model.add(BatchNormalization())

model.add(Dropout(0.6))



model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3), padding = "same"))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.6))



model.add(Conv2D(32, (3, 3), padding = "same"))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.6))



model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Flatten())

model.add(Dense(1028))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(total_classes))

model.add(Activation('softmax'))



model.compile(Adam(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])



model.summary()
model.fit(images_train, y_train, epochs=12, validation_data=(images_test, y_test), batch_size = 128)
#del history

del model



import gc

gc.collect()
base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
top_block = base_vgg16.output

top_block = GlobalAveragePooling2D()(top_block) # pool over height/width to reduce number of parameters

top_block = Dropout(0.5)(top_block)

top_block = Dense(256, activation='relu')(top_block) # add a Dense layer

top_block = Dropout(0.5)(top_block)

top_block = Dense(128, activation='relu')(top_block)

#top_block = Dropout(0.5)(top_block)

top_block = Dense(64, activation='relu')(top_block)

predictions = Dense(total_classes, activation='softmax')(top_block) # add another Dense layer



model_transfer = Model(inputs=base_vgg16.input, outputs=predictions)
model_transfer.summary()
#for layer in base_vgg19.layers:

#    layer.trainable = False
#model_transfer = None
model_transfer.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model_transfer.fit(images_train, y_train, validation_data=(images_test, y_test), epochs=15, batch_size = 128)
del model_transfer



import gc

gc.collect()
top_block = base_vgg16.output

predictions = GlobalAveragePooling2D()(top_block) # pool over height/width to reduce number of parameters



feature_extractor = Model(inputs=base_vgg16.input, outputs=predictions)



Z_train = feature_extractor.predict(images_train)

Z_test  = feature_extractor.predict(images_test)
y_train_label = []

for i in y_train:

    y_train_label.append(np.argmax(i))

    

y_test_label = []

for i in y_test:

    y_test_label.append(np.argmax(i))

    
lr = LogisticRegression(multi_class="multinomial", solver="lbfgs", C = 10, max_iter=2000)

lr.fit(Z_train, y_train_label)

print ("The training accuracy  for Lr is {}".format(lr.score(Z_train, y_train_label)))

print ("The testing accuracy  for Lr is {}".format(lr.score(Z_test, y_test_label)))