# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import os

import time

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten



from keras.applications.imagenet_utils import preprocess_input

from keras.layers import Input

from keras.models import Model

from keras.utils import np_utils

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
#img_path = os.path('../input/swith-on/dataset/')

img = image.load_img(r'../input/swithon/dataset/train/bad/augimg11 (2).jpg', target_size=(224, 224))

print(img)
x = image.img_to_array(img)

print (x.shape)

x = np.expand_dims(x, axis=0)

print (x.shape)

x = preprocess_input(x)

print('Input image shape:', x.shape)

PATH = '../'

# Define data path

data_path =  '../input/swithon/dataset/train/'

data_dir_list = os.listdir(data_path)



img_data_list=[]



for dataset in data_dir_list:

	img_list=os.listdir(data_path+'/'+ dataset)

	#print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

	for img in img_list:

		img_path = data_path + '/'+ dataset + '/'+ img 

		img = image.load_img(img_path, target_size=(224, 224))

		x = image.img_to_array(img)

		x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

		print('Input image shape:', x.shape)


# Loading the training data

PATH = '../'

# Define data path

data_path =  '../input/swithon/dataset/train'

data_dir_list = os.listdir(data_path)



img_data_list=[]

c=0

for dataset in data_dir_list:

	img_list=os.listdir(data_path+'/'+ dataset)

	#print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

	for img in img_list:

		img_path = data_path + '/'+ dataset + '/'+ img 

		img = image.load_img(img_path, target_size=(224, 224))

		x = image.img_to_array(img)

		x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

		#print('Input image shape:', x.shape)

		img_data_list.append(x)

		c+=1

	print(c)

print(img_list)

print(len(img_list))
img_data_list


img_data = np.array(img_data_list)

#img_data = img_data.astype('float32')

print (img_data.shape)

img_data=np.rollaxis(img_data,1,0)

print (img_data.shape)

img_data=img_data[0]

print (img_data.shape)
labels = np.asarray([1]*img_data.shape[0])

labels
num_classes = 2

num_of_samples = img_data.shape[0]

labels = np.asarray([1]*img_data.shape[0])



labels[0:117]=0

labels[117:227]=1
labels.shape
names = ['bad','good']

# convert class labels to on-hot encoding

Y = np_utils.to_categorical(labels, num_classes)
print(img_data)
#Shuffle the dataset

x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

image_input = Input(shape=(224, 224,3))



model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()



for layer in model.layers[:-1]:

   

        layer.trainable = False

  

model.layers[-1].trainable = True
last_layer = model.get_layer('avg_pool').output

#x= Flatten(name='flatten')(last_layer)

out = Dense(num_classes, activation='softmax', name='output_layer')(last_layer)

custom_resnet_model = Model(inputs=image_input,outputs= out)

custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:

	layer.trainable = False



custom_resnet_model.layers[-1].trainable



custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



t=time.time()


hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))

print('Training time: %s' % (t - time.time()))

(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)



print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

from sklearn.externals import joblib

joblib.dump(custom_resnet_model, 'custom_resnet_model.pkl')# model save at 91.6666% accuracy

model.save(custom_resnet_model.h5)
print(os.listdir("../input"))
# Define data path foe testing 

data_path =  '../input/swithon/dataset/validation/'

data_dir_list = os.listdir(data_path)



img_data_list=[]



for dataset in data_dir_list:

	img_list=os.listdir(data_path+'/'+ dataset)

	#print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

	for img in img_list:

		img_path = data_path + '/'+ dataset + '/'+ img 

		img = image.load_img(img_path, target_size=(224, 224))

		x = image.img_to_array(img)

		x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

		print('Input image shape:', x.shape)


# Loading the testing data

PATH = '../'

# Define data path

data_path =  '../input/swithon/dataset/validation'

data_dir_list = os.listdir(data_path)



img_data_list=[]

c=0

for dataset in data_dir_list:

	img_list=os.listdir(data_path+'/'+ dataset)

	#print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

	for img in img_list:

		img_path = data_path + '/'+ dataset + '/'+ img 

		img = image.load_img(img_path, target_size=(224, 224))

		x = image.img_to_array(img)

		x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

		print('Input image shape:', x.shape)

		img_data_list.append(x)

		c+=1

	print(c)


x_img_data_test = np.array(img_data_list)

#img_data = img_data.astype('float32')

print (x_img_data_test.shape)

x_img_data_test=np.rollaxis(img_data,1,0)

print (x_img_data_test.shape)

x_img_data_test=img_data[0]

print (x_img_data_test.shape)
x2=[]

for i in range(0,175):

    

    x = np.expand_dims(x_img_data_test, axis=0)

    x2.append(x)

#print(x)

for i in range(0,len(x2)):

    image_pred_y=custom_resnet_model.predict(x2[i], batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    #classes = custom_resnet_model.predict_classes(x)

    print(image_pred_y)



# Fine tune the resnet 50

#image_input = Input(shape=(224, 224, 3))

model = ResNet50(weights='imagenet',include_top=False)

model.summary()

last_layer = model.output

# add a global spatial average pooling layer

x = GlobalAveragePooling2D()(last_layer)

# add fully-connected & dropout layers

x = Dense(512, activation='relu',name='fc-1')(x)

x = Dropout(0.2)(x)

x = Dense(256, activation='relu',name='fc-2')(x)

x = Dropout(0.2)(x)

# a softmax layer for 2 classes

out = Dense(2, activation='softmax',name='output_layer')(x)
# this is the model we will train

custom_resnet_model2 = Model(inputs=model.input, outputs=out)



custom_resnet_model2.summary()


for layer in custom_resnet_model2.layers[:-6]:

	layer.trainable = False



custom_resnet_model2.layers[-1].trainable



custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



t=time.time()

hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=6, verbose=1, validation_data=(X_test, y_test))


print('Training time: %s' % (t - time.time()))

(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)



print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

print('Training time: %s' % (t - time.time()))

(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)



print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
import matplotlib.pyplot as plt

# visualizing losses and accuracy

train_loss=hist.history['loss']

val_loss=hist.history['val_loss']

train_acc=hist.history['accuracy']

val_acc=hist.history['val_accuracy']

xc=range(12)


###########################################################################################################################













############################################################################################





plt.figure(1,figsize=(7,5))

plt.plot(xc,train_loss)

plt.plot(xc,val_loss)

plt.xlabel('num of Epochs')

plt.ylabel('loss')

plt.title('train_loss vs val_loss')

plt.grid(True)

plt.legend(['train','val'])

#print plt.style.available # use bmh, classic,ggplot for big pictures

plt.style.use(['classic'])



plt.figure(2,figsize=(7,5))

plt.plot(xc,train_acc)

plt.plot(xc,val_acc)

plt.xlabel('num of Epochs')

plt.ylabel('accuracy')

plt.title('train_acc vs val_acc')

plt.grid(True)

plt.legend(['train','val'],loc=4)

#print plt.style.available # use bmh, classic,ggplot for big pictures

plt.style.use(['classic'])