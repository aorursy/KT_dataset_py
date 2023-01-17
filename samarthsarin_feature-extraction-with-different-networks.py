import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
# Extract features

import os, shutil

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

img_width = 224

img_height = 224

batch_size = 32

# Instantiate convolutional base



from keras.applications import ResNet50



conv_base = ResNet50(weights='imagenet',

                 include_top=False,

                 input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures

def extract_features(directory, sample_count):

   features = np.zeros(shape=(sample_count, 7, 7, 2048))  # Must be equal to the output of the convolutional base

   labels = np.zeros(shape=(sample_count,10))

   # Preprocess data

   generator = datagen.flow_from_directory(directory,

                                           target_size=(img_width,img_height),

                                           batch_size = batch_size,

                                           class_mode='categorical')

   # Pass data through convolutional base

   i = 0

   for inputs_batch, labels_batch in generator:

       features_batch = conv_base.predict(inputs_batch)

       features[i * batch_size: (i + 1) * batch_size] = features_batch

       labels[i * batch_size: (i + 1) * batch_size] = labels_batch

       i += 1

       if i * batch_size >= sample_count:

           break

   return features, labels



train_size = 1088

validation_size = 256
train_features, train_labels = extract_features('../input/10-monkey-species/training/training/', train_size)  # Agree with our small dataset size

validation_features, validation_labels = extract_features('../input/10-monkey-species/validation/validation/', validation_size)
features = np.concatenate((train_features, validation_features))
labels_train= []

for i in range(len(train_labels)):

    labels_train.append(np.argmax(train_labels[i])) 

    

labels_valid= []

for i in range(len(validation_labels)):

    labels_valid.append(np.argmax(validation_labels[i])) 
labels = np.concatenate((labels_train, labels_valid))
X_train, y_train = features.reshape(1344,7*7*2048),labels



x_train,x_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2,random_state = 42)



nb = MultinomialNB()

nb.fit(x_train, y_train)
pred = nb.predict(x_test)
accuracy_score(y_test,pred)
# Extract features

import os, shutil

import numpy as np

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)

img_width = 224

img_height = 224

batch_size = 32

# Instantiate convolutional base

from keras.applications import InceptionV3



conv_base = InceptionV3(weights='imagenet',

                 include_top=False,

                 input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures

def extract_features(directory, sample_count):

   features = np.zeros(shape=(sample_count, 5,5, 2048))  # Must be equal to the output of the convolutional base

   labels = np.zeros(shape=(sample_count,10))

   # Preprocess data

   generator = datagen.flow_from_directory(directory,

                                           target_size=(img_width,img_height),

                                           batch_size = batch_size,

                                           class_mode='categorical')

   # Pass data through convolutional base

   i = 0

   for inputs_batch, labels_batch in generator:

       features_batch = conv_base.predict(inputs_batch)

       features[i * batch_size: (i + 1) * batch_size] = features_batch

       labels[i * batch_size: (i + 1) * batch_size] = labels_batch

       i += 1

       if i * batch_size >= sample_count:

           break

   return features, labels



train_size = 1088

validation_size = 256
train_features, train_labels = extract_features('../input/10-monkey-species/training/training/', train_size)  # Agree with our small dataset size

validation_features, validation_labels = extract_features('../input/10-monkey-species/validation/validation/', validation_size)
features = np.concatenate((train_features, validation_features))
labels_train= []

for i in range(len(train_labels)):

    labels_train.append(np.argmax(train_labels[i])) 

    

labels_valid= []

for i in range(len(validation_labels)):

    labels_valid.append(np.argmax(validation_labels[i])) 
labels = np.concatenate((labels_train, labels_valid))
X_train, y_train = features.reshape(1344,5*5*2048), labels



x_train,x_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2,random_state = 42)



nb = MultinomialNB()

nb.fit(x_train, y_train)
pred = nb.predict(x_test)
accuracy_score(y_test,pred)
# Extract features

import os, shutil

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)

img_width = 224

img_height = 224

batch_size = 32

# Instantiate convolutional base

from keras.applications import VGG19



conv_base = VGG19(weights='imagenet',

                 include_top=False,

                 input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures

def extract_features(directory, sample_count):

   features = np.zeros(shape=(sample_count, 7,7,512))  # Must be equal to the output of the convolutional base

   labels = np.zeros(shape=(sample_count,10))

   # Preprocess data

   generator = datagen.flow_from_directory(directory,

                                           target_size=(img_width,img_height),

                                           batch_size = batch_size,

                                           class_mode='categorical')

   # Pass data through convolutional base

   i = 0

   for inputs_batch, labels_batch in generator:

       features_batch = conv_base.predict(inputs_batch)

       features[i * batch_size: (i + 1) * batch_size] = features_batch

       labels[i * batch_size: (i + 1) * batch_size] = labels_batch

       i += 1

       if i * batch_size >= sample_count:

           break

   return features, labels



train_size = 1088

validation_size = 256
train_features, train_labels = extract_features('../input/10-monkey-species/training/training/', train_size)  # Agree with our small dataset size

validation_features, validation_labels = extract_features('../input/10-monkey-species/validation/validation/', validation_size)
features = np.concatenate((train_features, validation_features))
labels_train= []

for i in range(len(train_labels)):

    labels_train.append(np.argmax(train_labels[i])) 

    

labels_valid= []

for i in range(len(validation_labels)):

    labels_valid.append(np.argmax(validation_labels[i])) 
labels = np.concatenate((labels_train, labels_valid))
X_train, y_train = features.reshape(1344,7*7*512), labels



x_train,x_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2,random_state = 42)



nb = MultinomialNB()

nb.fit(x_train, y_train)
pred = nb.predict(x_test)
accuracy_score(y_test,pred)
# Extract features

import os, shutil

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)

img_width = 224

img_height = 224

batch_size = 32

# Instantiate convolutional base

from keras.applications import InceptionResNetV2





conv_base = InceptionResNetV2(weights='imagenet',

                 include_top=False,

                 input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures



def extract_features(directory, sample_count):

   features = np.zeros(shape=(sample_count, 5,5,1536))  # Must be equal to the output of the convolutional base

   labels = np.zeros(shape=(sample_count,10))

   # Preprocess data

   generator = datagen.flow_from_directory(directory,

                                           target_size=(img_width,img_height),

                                           batch_size = batch_size,

                                           class_mode='categorical')

   # Pass data through convolutional base

   i = 0

   for inputs_batch, labels_batch in generator:

       features_batch = conv_base.predict(inputs_batch)

       features[i * batch_size: (i + 1) * batch_size] = features_batch

       labels[i * batch_size: (i + 1) * batch_size] = labels_batch

       i += 1

       if i * batch_size >= sample_count:

           break

   return features, labels



train_size = 1088

validation_size = 256
train_features, train_labels = extract_features('../input/10-monkey-species/training/training/', train_size)  # Agree with our small dataset size

validation_features, validation_labels = extract_features('../input/10-monkey-species/validation/validation/', validation_size)
features = np.concatenate((train_features, validation_features))
labels_train= []

for i in range(len(train_labels)):

    labels_train.append(np.argmax(train_labels[i])) 

    

labels_valid= []

for i in range(len(validation_labels)):

    labels_valid.append(np.argmax(validation_labels[i])) 
labels = np.concatenate((labels_train, labels_valid))
X_train, y_train = features.reshape(1344,5*5*1536), labels



x_train,x_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2,random_state = 42)



nb = MultinomialNB()

nb.fit(x_train, y_train)
pred = nb.predict(x_test)
accuracy_score(y_test,pred)
# Extract features

import os, shutil

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)

img_width = 224

img_height = 224

batch_size = 32

# Instantiate convolutional base

from keras.applications import MobileNetV2





conv_base = MobileNetV2(weights='imagenet',

                 include_top=False,

                 input_shape=(img_width, img_height, 3))  # 3 = number of channels in RGB pictures



def extract_features(directory, sample_count):

   features = np.zeros(shape=(sample_count, 7,7,1280))  # Must be equal to the output of the convolutional base

   labels = np.zeros(shape=(sample_count,10))

   # Preprocess data

   generator = datagen.flow_from_directory(directory,

                                           target_size=(img_width,img_height),

                                           batch_size = batch_size,

                                           class_mode='categorical')

   # Pass data through convolutional base

   i = 0

   for inputs_batch, labels_batch in generator:

       features_batch = conv_base.predict(inputs_batch)

       features[i * batch_size: (i + 1) * batch_size] = features_batch

       labels[i * batch_size: (i + 1) * batch_size] = labels_batch

       i += 1

       if i * batch_size >= sample_count:

           break

   return features, labels



train_size = 1088

validation_size = 256
train_features, train_labels = extract_features('../input/10-monkey-species/training/training/', train_size)  # Agree with our small dataset size

validation_features, validation_labels = extract_features('../input/10-monkey-species/validation/validation/', validation_size)
features = np.concatenate((train_features, validation_features))
labels_train= []

for i in range(len(train_labels)):

    labels_train.append(np.argmax(train_labels[i])) 

    

labels_valid= []

for i in range(len(validation_labels)):

    labels_valid.append(np.argmax(validation_labels[i])) 
labels = np.concatenate((labels_train, labels_valid))
X_train, y_train = features.reshape(1344,7*7*1280), labels



x_train,x_test,y_train,y_test = train_test_split(X_train,y_train,test_size = 0.2,random_state = 42)



nb = MultinomialNB()

nb.fit(x_train, y_train)
pred = nb.predict(x_test)
accuracy_score(y_test,pred)