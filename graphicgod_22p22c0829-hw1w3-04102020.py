# Import library

import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
train_path ='../input/super-ai-image-classification/train/train/images'
train_df = pd.read_csv('../input/super-ai-image-classification/train/train/train.csv')
def prepareImages(data, m, dataset):
    '''
    Function for preprocessing input
    '''
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['id']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/super-ai-image-classification/"+dataset+"/"+dataset+"/images"+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
def prepare_labels(y):
    '''
    Function for encoding label
    '''
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255
y, label_encoder = prepare_labels(train_df['category'])
# X.shape
# y.shape
# Create Train and Validation set
X_train = X[:1500]
X_val = X[1500:]
y_train = y[:1500]
y_val = y[1500:]
from sklearn.metrics import precision_score, recall_score
from keras import backend as K

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
'''
# Model 1: Normal CNN
model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

# model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# optimizer=tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=[f1_metric])
model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_val, y_val))
# Result 50 batched: 19s 1s/step - loss: 0.3681 - f1_metric: 0.8480 - val_loss: 0.8798 - val_f1_metric: 0.4300
'''
# Model 2: Transfer learning VGG16 (The best model)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(100,100, 3))

# fit output
x = layers.Flatten()(vgg.output)
x = layers.Dense(2, activation='softmax')(x)
model = keras.Model(vgg.input, x)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=[f1_metric])

# Fit Model
#history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_val, y_val))
history = model.fit(X, y, epochs=25, batch_size=100, verbose=1)

# Validation Result 24 Epoches 294s 20s/step - loss: 0.0706 - f1_metric: 0.9887 - val_loss: 0.6562 - val_f1_metric: 0.7700
'''
# Model 3: Transfer learning VGG16, metric = accuracy

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(100,100, 3))

# fit output
x = layers.Flatten()(vgg.output)
x = layers.Dense(2, activation='softmax')(x)
model = keras.Model(vgg.input, x)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics='accuracy')

# Fit Model
#history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_val, y_val))
history = model.fit(X, y, epochs=25, batch_size=100, verbose=1)

# Validation Result 24 Epoches 294s 20s/step - loss: 0.0706 - f1_metric: 0.9887 - val_loss: 0.6562 - val_f1_metric: 0.7700
'''
'''
# Model 4: Transfer learning ResNet50, metric = f1_metric --> Likely to overfit

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(100,100, 3))

# fit output
x = layers.Flatten()(resnet.output)
x = layers.Dense(2, activation='softmax')(x)
model = keras.Model(resnet.input, x)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=[f1_metric])

# Fit Model
# history = model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1, validation_data=(X_val, y_val))
history = model.fit(X, y, epochs=8, batch_size=100, verbose=1)

# Validation Result 24 Epoches 294s 20s/step - loss: 0.0706 - f1_metric: 0.9887 - val_loss: 0.6562 - val_f1_metric: 0.7700
'''
'''
# Model 5: Transfer learning VGG16adam optimizer


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(100,100, 3))

# fit output
x = layers.Flatten()(vgg.output)
x = layers.Dense(2, activation='softmax')(x)
model = keras.Model(vgg.input, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_metric])

# Fit Model
#history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_val, y_val))
history = model.fit(X, y, epochs=25, batch_size=100, verbose=1)
'''
'''
# Model 7: Transfer learning VGG16 change pic size to 225*225

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(100,100, 3))

# fit output
x = layers.Flatten()(vgg.output)
x = layers.Dense(2, activation='softmax')(x)
model = keras.Model(vgg.input, x)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=[f1_metric])

# Fit Model
#history = model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_val, y_val))
history = model.fit(X, y, epochs=25, batch_size=100, verbose=1)
'''
# Import test set
test = os.listdir("../input/super-ai-image-classification/val/val/images")
col = ['id']
test_df = pd.DataFrame(test, columns=col)
test_df['category'] = ''
X_test = prepareImages(test_df,test_df.shape[0], "val")
X_test /= 255
# Predict test set
predictions = model.predict(np.array(X_test), verbose=1)
test_df['category'] = predictions.argsort()[:,1]
test_df.head()
# Create a submission file
test_df.to_csv('submission.csv', index=False)