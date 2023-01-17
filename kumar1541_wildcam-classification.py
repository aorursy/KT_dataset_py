# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2

import keras

import keras.backend as K

from keras.models import Sequential

from keras.callbacks import Callback

from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,Activation

from glob import glob

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

%matplotlib inline
def vectorize(n):

    n = int(n)

    a = [0 for i in range(14)]

    if n <= 13:

        a[n] = 1

    else:

        a[0] = 1

    return a
X_train = np.load("../input/wildcam-reduced/X_train.npy")

Y_train = np.load("../input/wildcam-reduced/y_train.npy")
print(X_train.shape)

print(Y_train.shape)
X_test = np.load("../input/wildcam-reduced/X_test.npy")
X_train = X_train.astype("float32")/255.0

X_test = X_test.astype("float32")/255.0

# Y_train = Y_train.astype("float32")/255.0
# model1 = Sequential()



# model1.add(Conv2D(64,(3,3),strides = 3,padding = "same",activation = "relu",input_shape = (32,32,3)))

# model1.add(Conv2D(64,(3,3),strides = 3,padding = "same",activation = "relu"))

# model1.add(Dropout(0.35))



# model1.add(Conv2D(128,(3,3),strides = 3,padding = "same",activation = "relu"))

# model1.add(Dropout(0.35))





# model1.add(Conv2D(256,(3,3),strides = 3,padding = "same",activation = "relu"))





# model1.add(Conv2D(256,(3,3),strides = 3,padding = "same",activation = "relu"))





# model1.add(Conv2D(512,(3,3),strides = 3,padding = "same",activation = "relu"))

# model1.add(Dropout(0.20))



# model1.add(Flatten())

# model1.add(Dropout(0.20))

# model1.add(Dense(512,activation = "relu"))

# model1.add(Dropout(0.20))

# model1.add(Dense(128,activation = "relu"))

# model1.add(Dropout(0.20))

# model1.add(Dense(64,activation = "relu"))

# model1.add(Dropout(0.20))

# model1.add(Dense(32,activation = "relu"))

# model1.add(Dense(14,activation = "softmax"))

# model1.summary()















# model1.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
# model1.fit(X_train,Y_train,batch_size = 150,epochs = 20,validation_split = 0.25)


batch_size = 64

num_classes = 14

epochs = 30

val_split = 0.1

save_dir = os.path.join(os.getcwd(), 'models')

model_name = 'keras_cnn_model.h5'

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=X_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



hist = model.fit(

    X_train, 

    Y_train,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=val_split,

    shuffle=True

)

# vgg_light = Sequential()

# vgg_light.add(Conv2D(64,(3,3),padding = "same",strides = 3,activation = "relu",input_shape = (32,32,3)))

# vgg_light.add(Conv2D(64,(3,3),padding = "same",strides = 3,activation = "relu"))

# # vgg_light.add(MaxPooling2D(pool_size = (2,2)))

# vgg_light.add(Conv2D(128,(3,3),padding = "same",strides = 3,activation = "relu"))

# vgg_light.add(MaxPooling2D((2,2)))

# vgg_light.add(Conv2D(512,(3,3),padding = "same",strides = 3,activation = "relu"))

# # vgg_light.add(MaxPooling2D((2,2)))

# vgg_light.add(Dropout(0.3))

# vgg_light.add(Flatten())

# vgg_light.add(Dense(128,activation = "relu"))

# vgg_light.add(Dropout(0.2))

# vgg_light.add(Dense(64,activation = "relu"))

# vgg_light.add(Dropout(0.2))

# vgg_light.add(Dense(14,activation = "softmax"))

# # vgg_light.summary()
# vgg_light.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
# vgg_light.fit(X_train,Y_train,batch_size = 80,epochs = 30,validation_split = 0.25)
# vgg_light_predict = vgg_light.predict(X_test)

model_predict = model.predict(X_test)

# model1_predict = model1.predict(X_test)
# vgg_light_predict.shape
# vgg_light_predict[5]
submission_df = pd.read_csv('../input/iwildcam-2019-fgvc6/sample_submission.csv')

submission_df['Predicted'] = model_predict.argmax(axis=1)

matched_dict = {0: 0,

 1: 1,

 2: 3,

 3: 4,

 4: 8,

 5: 10,

 6: 11,

 7: 13,

 8: 14,

 9: 16,

 10: 17,

 11: 18,

 12: 19,

 13: 22}
submittable = model_predict

for i in range(submittable.shape[0]):

    submittable[i] = matched_dict[submittable[i]]
print(submission_df.shape)

submission_df.head()
submission_df.to_csv('submission_vgg_redefined.csv',index=False)
os.listdir()
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

# df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(submission_df)