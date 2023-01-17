# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import os

import pandas as pd

import numpy as np

import seaborn as sns



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D

from keras.utils import np_utils
#path of datasets

path_train = '../input/train.csv'

path_test = '../input/test.csv'
#create dataframe for training dataset and print 5 first rows as preview

train_df_raw = pd.read_csv(path_train)

train_df_raw.head()
# print infos about the dataset

train_df_raw.info()
train_df_raw.describe()
# Check if there are missing datas

train_df_raw.isnull().values.any()

train_df_raw.isna().values.any()
def dislay_images_from_pixels(nb): 

    images = [np.array(train_df_raw.drop(['label'], 1).iloc[i].tolist()).reshape(28, 28) for i in range (nb)]

    rows = int(nb/15) if int(nb/15) != 0 else int(nb/15) + 1

    plt.figure(figsize=(16, rows))

    for n in range(1, nb + 1):

        plt.subplot(rows, 15, n)

        plt.imshow(images[n-1])

        plt.axis('off')

        

dislay_images_from_pixels(120)
sns.set()

plt.figure(figsize=(20, 8))

sns.distplot(train_df_raw.label)

plt.show()
# Prepare data in order to pass it to a CNN

train_df = train_df_raw.copy()



# Separate target from images

X_train = train_df.drop(['label'], 1)

Y_train = train_df['label']



# Add 2 dimensions to pass a 4D tensor to the CNN and normalize values

X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 

# One-hot encoding of target

Y_train = np_utils.to_categorical(Y_train, 10)
def build_cnn():

    

    model = Sequential()

    

    # Multiple convolution operations to detect features in the images

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=3,activation='relu')) # no need to specify shape as there is a layer before

    model.add(BatchNormalization())

    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4)) # reduce overfitting



    model.add(Conv2D(64,kernel_size=3,activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size=3,activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4)) # reduce overfitting

    

    # Flattening and classification by standard ANN

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

    

    return model
build_cnn().summary()
X_test = pd.read_csv(path_test)

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
model = build_cnn()

model.fit(X_train, Y_train, batch_size=64, epochs=16)
prediction = model.predict_classes(X_test, verbose=0)

submission = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),

                         "Label": prediction})

submission.to_csv("submission.csv", index=False, header=True)