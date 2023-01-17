# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import InceptionV3

# from tensorflow.math import confusion_matrix

from tensorflow.contrib.metrics import confusion_matrix
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(data_df.shape)
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_df.shape
data_df['label'].value_counts().plot(kind='bar')
data_df.columns
X_data = data_df[data_df.columns[1:]]

Y_data = data_df['label']
print(X_data.columns[X_data.isnull().sum() > 0])

print(Y_data[Y_data.isnull()])
test_df.head()
# Normalize the data

X_data = X_data / 255.0

test_df = test_df / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_data = X_data.values.reshape(-1,28,28,1)

test_df = test_df.values.reshape(-1,28,28,1)
# Split the train and the validation set for the fitting

random_seed = 10

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size = 0.2, random_state=random_seed)

# X_train = X_train.reset_index(drop=True)

# X_val = X_val.reset_index(drop=True)

Y_val = Y_val.reset_index(drop=True)

Y_train = Y_train.reset_index(drop=True)

print(X_train.shape)

print(X_val.shape)
# pre_trained_model = InceptionV3(input_shape = (28, 28, 1), 

#                                 include_top = False, 

#                                 weights = None)

# for layer in pre_trained_model.layers:

#     layer.trainable = False
model = tf.keras.models.Sequential([

   tf.keras.layers.Conv2D(16, (3,3), strides=(1,1), activation='relu', 

                           input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    

    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



#     tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(120, activation='relu'),

    tf.keras.layers.Dense(84, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])

# Y_train = tf.keras.utils.to_categorical(Y_train, 10)

# Y_val = tf.keras.utils.to_categorical(Y_val, 10)
# # performing data augmentation

# datagen = ImageDataGenerator(

#     rotation_range=40,

#     width_shift_range=0.2,

#     height_shift_range=0.2)

# # compute quantities required for featurewise normalization

# # (std, mean, and principal components if ZCA whitening is applied)

# datagen.fit(X_train)
model.compile(optimizer='rmsprop',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.summary()
# history = model.fit_generator(datagen.flow(X_train, Y_train), epochs=20, validation_data=(X_val, Y_val))

history = model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val))
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)
matrix = confusion_matrix(Y_val, pred_val)

with tf.Session():

   print('Confusion Matrix: \n\n', tf.Tensor.eval(matrix,feed_dict=None, session=None))
pred_val =  model.predict(X_val)

# find the class with the highest probability

pred_val = pd.Series(np.argmax(pred_val,axis = 1))

res = pred_val - Y_val

missclassified = [False if item==0 else True for item in res]

mis = X_val[missclassified]

i = 0

limit = 10

mis.shape
for i in range(20):

    plt.figure()

    plt.imshow(np.resize(mis[i+20], (28, 28)))
model.save('model.h5')
# make prediction, gives each class (0,9) a probability

results = model.predict(test_df)

# find the class with the highest probability

results = pd.Series(np.argmax(results,axis = 1))

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("sumbission.csv",index=False)
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(submission)
Y_train[12]