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
data_original_test= pd.read_csv("/kaggle/input/fashion-apparel-identification/test_ScVgIM0/test.csv")

data_original_train=pd.read_csv("/kaggle/input/fashion-apparel-identification/train_LbELtWX/train.csv")
## checking datasets

data_original_train.head(1000)
data_original_test.tail(100)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm
#read all the training images, store them in a list, and finally convert that list into a numpy array.

# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False

train_image = []

for i in tqdm(range(data_original_train.shape[0])):

    img = image.load_img('/kaggle/input/fashion-apparel-identification/train_LbELtWX/train/'+data_original_train['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)
#As it is a multi-class classification problem (10 classes), we will one-hot encode the target variable.

y=data_original_train['label'].values

y = to_categorical(y)



#Creating a validation set from the training data.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
#model1

#create a simple architecture with 2 convolutional layers, one dense hidden layer and an output layer.

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



#modification1 : adding more dropout layers (accuracy decreased)

#modification2 : using ResNet50
#compile the model created.

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
##Training the model : train the model on the training set images and validate it using the validation set.



model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
#Load the test images and predict their classes using the model.predict_classes() function.

#read and store all the test images:



test_image = []

for i in tqdm(range(data_original_test.shape[0])):

    img = image.load_img('/kaggle/input/fashion-apparel-identification/test_ScVgIM0/test/'+data_original_test['id'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)

    img = image.img_to_array(img)

    img = img/255

    test_image.append(img)

test = np.array(test_image)
# making predictions

prediction = model.predict_classes(test)

prediction
submission= pd.concat([data_original_test['id'],pd.DataFrame(prediction,columns=['label'])],axis=1)

submission
# creating submission file

submission.to_csv('submission3_apparel.csv', header=True, index=False)