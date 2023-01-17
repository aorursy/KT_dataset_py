# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing Libraries

import numpy as np

import pandas as pd

import pickle

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,GlobalMaxPooling2D,BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.applications.inception_v3 import InceptionV3

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import os

import random

import zipfile

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,precision_score,recall_score

#Change the working directory

os.chdir('/kaggle/working/')
#Extracting Files

train = "../input/dogs-vs-cats/train.zip"

test = "../input/dogs-vs-cats/test1.zip"

with zipfile.ZipFile(train,'r') as z:

    z.extractall('.')

with zipfile.ZipFile(test, 'r') as z:

    z.extractall('.')
#Making a DataFrame

filename = os.listdir("/kaggle/working/train")

categories = []

for name in filename:

    category = name.split(".")[0]

    if(category=='cat'):

        categories.append(0)

    else:

        categories.append(1)



data = pd.DataFrame({

    'filename':filename,

    'category':categories

})

        

print(data.head())       
#Inception Model (Pre-Trained)

local_weights_file = "../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = InceptionV3(input_shape=(150,150,3),include_top=False,weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:

  layer.trainable=False

pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')

last_output = last_layer.output

x = GlobalMaxPooling2D()(last_output)

x = Dense(1024,activation='relu')(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x)

x = Dense(1,activation='sigmoid')(x) 



model = Model(pre_trained_model.input,x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])

model.summary()
#Split into train and validation data

data['category'] = data['category'].replace({0:'cat',1:'dog'})

train_data,val_data = train_test_split(data,test_size=0.1,random_state=1)

train_data = train_data.reset_index(drop=True)

val_data = val_data.reset_index(drop=True)
#Generator

train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,

                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(train_data,

                                                    directory= "/kaggle/working/train/",

                                                    class_mode='binary',

                                                    target_size=(150,150),

                                                    x_col="filename",

                                                    y_col="category",

                                                    batch_size=32)



val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_dataframe(val_data,

                                                directory= "/kaggle/working/train/",

                                                class_mode='binary',

                                                target_size=(150,150),

                                                x_col="filename",

                                                y_col="category",

                                                batch_size=32)
#Callbacks

earlystop = EarlyStopping(monitor='val_loss',patience=4,verbose=1)

learning_reduce = ReduceLROnPlateau(patience=2,monitor="val_acc",verbose=1,min_lr=0.00001,factor=0.5)

callbacks = [earlystop,learning_reduce]
#Fitting the model

history = model.fit(train_generator,

                    validation_data = val_generator,

                    steps_per_epoch = len(train_data)//32,

                    validation_steps = len(val_data)//32,

                    epochs=10,

                    callbacks=callbacks)
#Visualize Training

def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history["val_"+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string,"val_"+string])

    plt.show()

plot_graphs(history,'acc')

plot_graphs(history,'loss')
#Save Model

model.save_weights('binary_classifier.h5')
#Accuracy and Loss of Validation Data

loss,accuracy = model.evaluate_generator(val_generator,steps=np.ceil(len(val_data)/32),verbose=1)

print("Validation Accuracy: ",accuracy)

print("Validation Loss: ",loss)
#Getting Predicted Value

y_val = val_data['category'].replace({'cat':0,'dog':1})

y_pred = model.predict_generator(val_generator,steps=np.ceil(len(val_data)/32))

y_final = y_pred.round().astype(int)
#Confusion Matrix

confusion = confusion_matrix(y_val, y_final) 



sns.heatmap(confusion, annot=True,cmap="Blues",fmt='.1f')

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
#Test Data

test_filename=os.listdir("/kaggle/working/test1")

test_data = pd.DataFrame({'filename': test_filename})
#Test Generator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(test_data,

                                                directory= "/kaggle/working/test1/",

                                                class_mode=None,

                                                target_size=(128,128),

                                                x_col="filename",

                                                y_col=None,

                                                batch_size=32)
#Predict

predict= model.predict_generator(test_generator,steps=np.ceil(len(test_data)/32),verbose=1)

test_data['category'] = np.where(predict>0.5,1,0)
#Visualize predicted resuls with images

sample_test = test_data.sample(n=9).reset_index()

sample_test.head()

plt.figure(figsize=(12,12))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img("./test1/"+filename, target_size=(150,150))

    plt.subplot(3, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')')

plt.tight_layout()

plt.show()