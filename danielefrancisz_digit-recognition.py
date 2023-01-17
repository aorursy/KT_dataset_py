import numpy as np

import pandas as pd



# Loading the images



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Create training data matrix



train_file = "../input/digit-recognizer/train.csv"

train_data = np.loadtxt(train_file, skiprows=1, delimiter=',')



print(train_data)

print(train_data.shape)
from tensorflow import keras



# Preparing and normalizing the data for training



num_classes = 10

num_imgs = train_data.shape[0]



img_rows = 28

img_columns = 28



y_raw = train_data[:,0]

y_train = keras.utils.to_categorical(y_raw, num_classes)



x_raw = train_data[:,1:]

x_train = x_raw.reshape(num_imgs, img_rows, img_columns,1)

x_train = x_train/255



print(x_raw)

print(y_train)

print(num_imgs)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D



# We create the model

# It is advised to add Dense layers after flattening.



my_model = Sequential()

my_model.add(Conv2D(

    filters=20,

    kernel_size=3,

    input_shape=(img_rows, img_columns, 1),

    activation='relu',

))



my_model.add(Conv2D(

    filters=20,

    kernel_size=3,

    input_shape=(img_rows, img_columns, 1),

    activation='relu',

))



my_model.add(Flatten())

my_model.add(Dense(units=100,activation='relu'))

my_model.add(Dense(units=10,activation='softmax'))



my_model.compile(

    optimizer = 'adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'],

)

# We fit the model and check its expected accuracy



my_model.fit(

    x=x_train,

    y=y_train,

    batch_size=100,

    epochs=4, 

    verbose=1,

    validation_split=0.2,

)
# Preparing the test data



num_test_imgs = 28000



test_file = "../input/digit-recognizer/test.csv"

test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')

print(test_data.shape)



X_raw = test_data[:,0:]

print(X_raw.shape)



X_test = X_raw.reshape(num_test_imgs, img_rows, img_columns,1)

X_test = X_test/255
# Getting predictions from test data



pred = my_model.predict(X_test)

print('Predictions',pred)
# Prepare predictions for submission



i = 1

pred_dict = {'ImageId':[],'Label':[]}

for image_row in pred:

    index_max = max(range(len(image_row)), key=image_row.__getitem__)

    pred_dict['ImageId'].append(i)

    pred_dict['Label'].append(index_max)

    i=i+1

    

submission = pd.DataFrame(data=pred_dict)



submission.tail()
submission.to_csv('submission.csv', index=False)