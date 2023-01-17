import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import os

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.models import Model



from tqdm import tqdm

import cv2



from keras.applications.vgg19 import VGG19

from keras.layers import Dense, Flatten,Dropout

from keras.models import Sequential



from keras.layers import (AveragePooling2D, Convolution2D, Dense, Dropout, Flatten, Input, MaxPooling2D) 

from keras.models import Model, Sequential



print(os.listdir("../input/"))
df_train = pd.read_csv('../input/dog-breed-identification/labels.csv')

train= ('../input/dog-breed-identification/train')

df_test = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')

test= ('../input/dog-breed-identification/test')

resnet_weight_paths=("../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
df_test.head()
df_train.head()
len(df_train["breed"].unique())
num_classes = 120
target_labels = df_train["breed"]

one_hot = pd.get_dummies(target_labels, sparse = True)

one_hot_labels = np.asarray(one_hot)
demo = ["a","b","c","d","e"]

demo_df = pd.DataFrame(demo)

print(demo_df)

one_hot_demo = pd.get_dummies(demo_df,sparse = True)

one_hot_demo = np.asarray(one_hot_demo)

one_hot_demo
img_size = 90

x_train=[]

y_train=[]

i=0

for f in tqdm(df_train.id):

    img = cv2.imread('../input/dog-breed-identification/train/{0}.jpg'.format(f))

    label = one_hot_labels[i]

    x_train.append(cv2.resize(img, (img_size, img_size)))

    y_train.append(label)

    i += 1
x_test = []

for f in tqdm(df_test.id):

    img = cv2.imread('../input/dog-breed-identification/test/{0}.jpg'.format(f))

    x_test.append(cv2.resize(img, (img_size,img_size))) 
x_train_raw = np.array(x_train).astype("float32")/255.
y_train_raw = np.array(y_train).astype("uint8")
x_test = np.array(x_test, np.float32) / 255.
print(x_train_raw.shape)

print(y_train_raw.shape)

print(x_test.shape)
train_X,test_X,train_Y,test_Y = train_test_split(x_train_raw,y_train_raw,test_size = 0.3, random_state=3)
base_model = VGG19(weights = None, include_top=False, input_shape=(img_size, img_size, 3))



# Add a new top layer

x = base_model.output

x = Flatten()(x)

predictions = Dense(num_classes, activation='softmax')(x)



# This is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# First: train only the top layers (which were randomly initialized)

for layer in base_model.layers:

    layer.trainable = False

    

model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

#Take a look at the architecture.

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

model.summary()
model.fit(train_X, train_Y, epochs=1, validation_data=(test_X, test_Y), verbose=1)
predictions = model.predict(x_test)
weights = "../input/resnet50/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"

new_model = vgg19(weights = weights, include_top=False, input_shape=(img_size, img_size, 3))

new_model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

new_model.fit(train_X, train_Y, epochs=1, validation_data=(test_X, test_Y), verbose=1)

new_predicts = new_model.predict 
#sub = pd.DataFrame(new_predict)

# Use the previous one_hot_encoded values to give the name to the columns

#col_names = one_hot.columns.values

#sub.columns = col_names



#sub.insert(0, 'id', df_test['id'])

#sub.head(5)