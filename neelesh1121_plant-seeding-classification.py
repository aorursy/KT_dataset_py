import numpy as np #supporting multi-dimensional arrays and matrices

import os #read or write a file

import cv2  

import pandas as pd #data manipulation and analysis

from tqdm import tqdm # for  well-established ProgressBar



from random import shuffle #only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.
data_dir = '../input/plant_seeding_dataset/home/neelesh/Documents/plant_seeding'

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

IMG_SIZE = 224
#list of categories in array format 

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

NUM_CATEGORIES = len(CATEGORIES)

print (NUM_CATEGORIES)
#OneHotEncoding

def label_img(word_label):                       

    if word_label == 'Black-grass': return [1,0,0,0,0,0,0,0,0,0,0,0]

    elif word_label == 'Charlock': return [0,1,0,0,0,0,0,0,0,0,0,0]

    elif word_label == 'Cleavers': return [0,0,1,0,0,0,0,0,0,0,0,0]

    elif word_label == 'Common Chickweed': return [0,0,0,1,0,0,0,0,0,0,0,0]

    elif word_label == 'Common wheat': return [0,0,0,0,1,0,0,0,0,0,0,0]

    elif word_label == 'Fat Hen': return [0,0,0,0,0,1,0,0,0,0,0,0]

    elif word_label == 'Loose Silky-bent': return [0,0,0,0,0,0,1,0,0,0,0,0]

    elif word_label == 'Maize': return [0,0,0,0,0,0,0,1,0,0,0,0]

    elif word_label == 'Scentless Mayweed': return [0,0,0,0,0,0,0,0,1,0,0,0]

    elif word_label == 'Shepherds Purse': return [0,0,0,0,0,0,0,0,0,1,0,0]

    elif word_label == 'Small-flowered Cranesbill': return [0,0,0,0,0,0,0,0,0,0,1,0]

    elif word_label == 'Sugar beet': return [0,0,0,0,0,0,0,0,0,0,0,1] 
#to create the test data

def create_train_data():

    train = []

    for category_id, category in enumerate(CATEGORIES):

        for img in tqdm(os.listdir(os.path.join(train_dir, category))):

            label=label_img(category)

            path=os.path.join(train_dir,category,img)

            img=cv2.imread(path)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

            train.append([np.array(img),np.array(label)])

    shuffle(train)

    return train

#train data

train_data = create_train_data()
#to create the test data

test = []

def create_test_data():

    for img in tqdm(os.listdir(test_dir)):

        path = os.path.join(test_dir,img)

        img_num = img

        img = cv2.imread(path)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        test.append([np.array(img), img_num])

        

    shuffle(test)

    return test

#test data

test_data = create_test_data()
#splitting the labels and features

x_train = []

y_train = []

for features, labels in train_data:

    x_train.append(features)

    y_train.append(labels)
x_train = np.array(x_train).reshape(-1, 224,224,3)
np.array(y_train).shape
import matplotlib.pyplot as plt

plt.imshow(x_train[0])

plt.show()
#splitting the features and labels

x_test = []

y_test = []

for features, labels in test_data:

    x_test.append(features)

    y_test.append(labels)
x_test = np.array(x_test).reshape(-1, 224,224,3)

import matplotlib.pyplot as plt

plt.imshow(x_test[0])

plt.show()
#For data augmentation

CATEGORIES = ['Common wheat', 'Maize', 'Cleavers', 'Black-grass', 'Shepherds Purse']

IMG_SIZE = 224

train = []

rows = 224

cols = 224

def create_augment_train_data():

    for category_id, category in enumerate(CATEGORIES):

        i = 0

        for img in tqdm(os.listdir(os.path.join(train_dir, category))):

            label=label_img(category)

            path=os.path.join(train_dir,category,img)

            img=cv2.imread(path)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

            M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)

            img = cv2.warpAffine(img,M,(cols,rows)) 

            train.append([np.array(img),np.array(label)])

            i = i+1

            if i==100:

                break

    shuffle(train)

    return train
augment_train_data = create_augment_train_data()
#splitting the augment data into features and labels

x_augment_train = []

y_augment_train = []

for features, labels in augment_train_data:

    x_augment_train.append(features)

    y_augment_train.append(labels)
x_augment_train = np.array(x_augment_train).reshape(-1, 224,224,3)
import matplotlib.pyplot as plt

plt.imshow(x_augment_train[0])

plt.show()
train_data = np.append(x_train, x_augment_train, axis = 0)
x_test_data = train_data[:500]

x_train_data = train_data[500:]

x_test_data.shape
x_train_data.shape
train_data = np.append(y_train, y_augment_train, axis = 0)


y_test_data = train_data[:500]

y_train_data = train_data[500:]

y_test_data.shape
#VGG16 Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras.layers import Input, Flatten, Dense, Conv2D, Dropout, MaxPooling2D, BatchNormalization

from keras.models import Model

from keras.callbacks import ModelCheckpoint

import numpy as np



image_input = Input(shape=(224,224,3))

vgg_mod = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

for layer in vgg_mod.layers[:7]:

    layer.trainable = False



vgg_mod.summary()

#defining the model

last_layer = vgg_mod.get_layer('block5_pool').output

x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(last_layer)

x = BatchNormalization(momentum = 0.95,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = Dense(4096, activation='relu')(x)

x = Dropout(0.4)(x)

x = Dense(128, activation='relu')(x)

out = Dense(12,activation='softmax',name='output')(x)
cust_vgg_model = Model(image_input,out)

cust_vgg_model.summary()
#to compile the model

import keras

keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



cust_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

#run the model

hist_1=cust_vgg_model.fit(x_train_data,y_train_data,epochs=10, verbose=1)
cust_vgg_model.save('cust_vgg_model.h5')
#plotting loss vs epochs

import matplotlib.pyplot as plt

print(hist_1.history.keys())



plt.plot(hist_1.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
#Simple CNN architecture 

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ReduceLROnPlateau

#earlystop the model on the basis of validation loss

es = EarlyStopping(monitor='val_loss', mode='min',verbose  = 1, patience = 5)



#to reduce the learning if accuracy is not getting bettter after 5 epochs

rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1E-7)

#simple Convolution Layer 

from keras.layers import Input, Flatten, Dense, Convolution2D, Dropout, MaxPooling2D, BatchNormalization

from keras.models import Sequential

import numpy as np

classifier = Sequential()

classifier.add(Convolution2D(64,3,3, input_shape = (224,224,3),activation = 'relu'))

classifier.add(BatchNormalization(momentum = 0.9,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32,3,3,activation = 'relu'))

classifier.add(BatchNormalization(momentum = 0.9,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 256, activation = 'relu'))

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim = 12, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
hist_3 = classifier.fit(x_train_data,y_train_data,epochs=10,validation_split = 0.2)
classifier.save('CNN_Model.h5')
#plotting loss vs epoches 

import matplotlib.pyplot as plt

plt.plot(hist_3.history['loss'])

plt.plot(hist_3.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#Inception V3 Model

from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization

from keras import backend as K



base_model = InceptionV3(weights='imagenet', include_top=False)



x = base_model.output

x = BatchNormalization(momentum = 0.9,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')(x)

x = GlobalAveragePooling2D()(x)



x = Dense(1024, activation='relu')(x)



predictions = Dense(12, activation='softmax')(x)



inception_model = Model(inputs=base_model.input, outputs=predictions)

inception_model.summary()
inception_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

hist_2=inception_model.fit(x_train_data,y_train_data,epochs=10,verbose=1,validation_split = 0.1)
inception_model.save('inception_model.h5')
# summarize history for loss

import matplotlib.pyplot as plt

plt.plot(hist_2.history['loss'])

plt.plot(hist_2.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#Resnet Model

import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50

from keras.applications.resnet50 import preprocess_input

from keras import Model, layers

from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization

from keras.models import load_model, model_from_json

from keras.callbacks import ModelCheckpoint



conv_base = ResNet50(include_top=False, weights='imagenet')



x = conv_base.output

x = BatchNormalization(momentum = 0.9,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')(x)

x = GlobalAveragePooling2D()(x)



x = Dense(1024, activation='relu')(x)

x = Dense(256, activation='relu')(x)



predictions = Dense(12, activation='softmax')(x)



resnet_model = Model(inputs=conv_base.input, outputs=predictions)

resnet_model.summary()


resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])
hist_4=resnet_model.fit(x_train_data,y_train_data,epochs=10,validation_split = 0.1, verbose=1)


resnet_model.save('resnet_model.h5')
# summarize history for loss

import matplotlib.pyplot as plt

plt.plot(hist_4.history['loss'])

plt.plot(hist_4.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#test the resnet model on test data

from keras.models import load_model

from sklearn.metrics import confusion_matrix

resnet_model = load_model('resnet_model.h5')

resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

test_loss, test_accuracy = resnet_model.evaluate(x_test_data,y_test_data)

confusion_matrix = (y_test_data, resnet_model.predict(x_test_data))

print(test_loss, ',',test_accuracy)

print('confusion matrix = ', confusion_matrix)
#test the inception model on test data

from sklearn.metrics import confusion_matrix

from keras.models import load_model

inception_model = load_model('inception_model.h5')

inception_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

test_loss, test_accuracy = inception_model.evaluate(x_test_data,y_test_data)

confusion_matrix = (y_test_data, inception_model.predict(x_test_data))

print(test_loss, ',',test_accuracy)

print('confusion matrix = ', confusion_matrix)
#test the VGG model on test data

from sklearn.metrics import confusion_matrix

from keras.models import load_model

cust_vgg_model = load_model('cust_vgg_model.h5')

cust_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

test_loss, test_accuracy = cust_vgg_model.evaluate(x_test_data,y_test_data)

confusion_matrix = (y_test_data, cust_vgg_model.predict(x_test_data))

print(test_loss, ',',test_accuracy)

print('confusion matrix = ', confusion_matrix)
#test the CNN model on test data

from sklearn.metrics import confusion_matrix

from keras.models import load_model

CNN_Model = load_model('CNN_Model.h5')

CNN_Model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

test_loss, test_accuracy = CNN_Model.evaluate(x_test_data,y_test_data)

confusion_matrix = (y_test_data, CNN_Model.predict(x_test_data))

print(test_loss, ',',test_accuracy)

print('confusion matrix = ', confusion_matrix)