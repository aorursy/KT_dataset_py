# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models, layers

from keras.optimizers import Adam

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from keras.utils import to_categorical

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')



#separate labels from image data

train_labels = train_data["label"]

train_data = train_data.drop(['label'], axis = 1)



#normalize data

train_data = train_data /255.

test_data = test_data / 255.



#reshape the data for CNNs

train_data = train_data.values.reshape(train_data.shape[0],28,28,1)

test_data = test_data.values.reshape(test_data.shape[0],28,28,1)



#convert labels to one-hot

labels = to_categorical(train_labels)



#split into test and training

X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size = 0.2, random_state = 42)
learning_rate_cb = ReduceLROnPlateau(monitor = 'val_acc', 

                                     patience = 2, 

                                     verbose = 1, 

                                     factor = 0.5, 

                                     min_lr = 1e-5)
datagen = ImageDataGenerator(

    zoom_range = 0.1,

    width_shift_range = 0.1,

    height_shift_range = 0.1, 

    rotation_range = 10

)



datagen.fit(X_train)
batchsize = 512

num_epochs = 30

n_model_runs = 10

modellist = list()



for i in range(n_model_runs):

    print("+++++++++  running model number", i+1)

    

    model = models.Sequential([

        Conv2D(16, [5,5], activation = 'relu', padding = 'same', input_shape = [28,28,1]),

        MaxPooling2D([2,2]),

        Conv2D(32, [5,5], activation = 'relu', padding = 'same'),

        MaxPooling2D([2,2]),

        Conv2D(64, [3,3], activation = 'relu', padding = 'same'),

        MaxPooling2D([2,2]),

        Conv2D(64, [3,3], activation = 'relu', padding = 'same'),

        MaxPooling2D([2,2]),

        Flatten(),

        Dense(512, activation = 'relu'),

        Dropout(0.3),

        Dense(1024, activation = 'relu'),

        Dropout(0.5),

        Dense(10, activation = 'softmax')

    ])



    model.compile(optimizer = Adam(lr=1e-3), 

                  loss='categorical_crossentropy',

                  metrics = ['accuracy'])



    #This is the model fit function without data augmentation for reference

    #history = model.fit(x = X_train, 

    #                    y = y_train, 

    #                    batch_size = batchsize, 

    #                    epochs = num_epochs, 

    #                    validation_data=(X_test,y_test),

    #                   callbacks=[learning_rate_cb])

    

    model.fit_generator(datagen.flow(X_train, y_train, batchsize),

                         epochs = num_epochs,

                         steps_per_epoch = X_train.shape[0]/batchsize,

                         validation_data = (X_test, y_test),

                         callbacks=[learning_rate_cb],

                         verbose = 1)

    

    modellist.append(model)
prediction = [model.predict(test_data) for model in modellist]

prediction = np.sum(prediction, axis=0)

prediction = np.argmax(prediction,axis=1)
#submit results

submission = pd.DataFrame({"ImageId": list(range(1, len(prediction)+1)), "Label": prediction})

submission.to_csv('submission.csv', index = False, header = True)