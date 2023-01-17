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
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import scipy.misc

from keras.layers import Dense,Dropout,Flatten

from keras.layers.convolutional import Convolution2D,MaxPooling2D,Conv2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD,Adadelta,Adagrad,Adam,Adamax

from keras.utils import np_utils

from keras.models import Sequential

from keras import models,initializers

import  keras.callbacks as kcallbacks



base_path = "/kaggle/input/digit-recognizer/"

def baseline_model():

    #create model

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),border_mode='valid',activation='relu',kernel_initializer='he_normal'))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.20))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.20))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128,activation='relu',kernel_initializer=initializers.TruncatedNormal(mean=0, stddev=0.001)))

    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu', kernel_initializer=initializers.TruncatedNormal(mean=0, stddev=0.001)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    # model.add(Dense(10,activation='relu',kernel_initializer=initializers.TruncatedNormal(mean=0, stddev=0.001)))

    model.add(Dense(num_classes,activation='softmax'))

    #Compile model

    model.compile(loss='categorical_crossentropy',optimizer=Adadelta(0.5),metrics=['accuracy'])

    return model
train_data = pd.read_csv(base_path+"train.csv",header=0)

label = np_utils.to_categorical(np.array(train_data["label"]))

feature = (np.array(train_data.iloc[:,1:])/255).reshape(train_data.shape[0],28,28,1)

num_classes = label.shape[1]

num = feature.shape[0]
data_label =[]

for i in range(5):

    data_label += [i]*(int(num*(i+1)/5)-int(num*i/5))

data_label=np.array(data_label)



for i in range(5):

    X_train = feature[data_label!=i]

    y_train = label[data_label!=i]

    X_test = feature[data_label == i]

    y_test = label[data_label == i]

    best_weights_filepath = base_path+"my_model_%d.h5"%i

    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,

                                               save_best_only=True, mode='auto')

    model = baseline_model()

    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=500,batch_size=200,verbose=1,callbacks=[earlyStopping, saveBestModel])

from keras import models

import pandas as pd

import numpy as np



pred_data = pd.read_csv(base_path+"test.csv",header=0)

preddata = (np.array(pred_data)/255).reshape(pred_data.shape[0],28,28,1)

final_data = []

for i in range(5):

    model = models.load_model("my_model_%d.h5"%i)

    final_data += [model.predict(preddata)]



final_data_np = np.array(final_data)

finaldata = final_data_np.mean(0)

a=pd.DataFrame({"ImageId":range(1,28001),"Label":finaldata.argmax(1)})

a.to_csv(base_path+"sample_submission.csv",index=False)