# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Suppress multiple warnings

import warnings

warnings.filterwarnings(action='once')



from keras.models import Sequential  # Model

from keras.layers import BatchNormalization, Conv2D , MaxPooling2D  #Layers used

from keras.layers import Lambda, Flatten, Dropout, Dense

from keras.preprocessing import image as px_image      # Image generator 



import matplotlib.pyplot as plt # Plotting inline

%matplotlib inline    

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Set the Root directory

root_dir = '/kaggle/input/digit-recognizer'

os.listdir(root_dir)
train_df = pd.read_csv(os.path.join(root_dir,'train.csv'))

print(train_df.shape)

# view training data head

train_df.head()
test_df = pd.read_csv(os.path.join(root_dir,'test.csv'))

print(test_df.shape)

# view test data head

test_df.head()
#extract Label from Training data

Y_train = train_df.label

#turn label to numpy list of float32

Y_train = (Y_train.values).astype('float32')



print(Y_train)



#Drop label from input data

X_train = train_df.drop('label', axis=1)

#turn data to numpy list of float32

X_train = (X_train.values).astype('float32')



print(X_train)

data_show_index = 3

# show data by reshaping numpy array to 28X28 for a image array

plt.imshow(X_train[data_show_index].reshape(28,28), cmap=plt.get_cmap('gray'))



# reshape the training input data to 28x28 image for 2D convolution operations

# shape with full batch of image with img.shape[0] for batching inputs

X_train = X_train.reshape(X_train.shape[0],28,28,1)

print(X_train.shape)
# turn the test data to numpy array with float32 type and correct shape

X_test = (test_df.values).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

print(X_test.shape)
datas_mean = X_train.mean().astype(np.float32)

datas_std = X_train.std().astype(np.float32)



#standardize data value to be centered around 0.

def standardize(data): 

    return (data-datas_mean)/datas_std

from keras.utils.np_utils import to_categorical



# label is categorical data

Y_train= to_categorical(Y_train)

num_classes = Y_train.shape[1]

print(num_classes)
# fix random seed for reproducibility

np.random.seed(42)
model = Sequential()

model.add(Lambda(standardize, input_shape=(28, 28, 1)))

model.add(Conv2D(16, 3, activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, 5, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))



print("model input shape ",model.input_shape)

print("model output shape ",model.output_shape)

# Evaluvate Model structure

model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
Image_generator = px_image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05, shear_range=0.2,

                                           height_shift_range=0.05, zoom_range=0.03)
BATCH_SIZE = 64

from sklearn.model_selection import train_test_split

X = X_train

Y = Y_train

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)

train_batches = Image_generator.flow(X_train, Y_train, batch_size=BATCH_SIZE)

val_batches = Image_generator.flow(X_val, Y_val, batch_size=BATCH_SIZE)
history = model.fit_generator(generator=train_batches, steps_per_epoch=train_batches.n, epochs=5, 

                              validation_data=val_batches, validation_steps=val_batches.n) 
history_dict = history.history

history_dict.keys()
epoch = [1,2,3,4,5]

plt.plot(epoch,history_dict['acc'], label='Acc')

plt.plot(epoch,history_dict['val_acc'], label='val_acc')

plt.legend()
predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)