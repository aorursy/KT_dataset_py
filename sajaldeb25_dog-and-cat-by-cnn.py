# https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial



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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import cv2

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import cv2

main_dir = "../input/"

train_dir = "train/train"

path = os.path.join(main_dir, train_dir)



for p in os.listdir(path):

    category = p.split(".")[1]

    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE )

    new_img_array = cv2.resize(img_array,dsize=(90,90))

    plt.imshow(new_img_array,cmap="gray")

    break
X = []

Y = []



convert = lambda category : int(category == 'dog')



def create_test_data(path):

    for p in os.listdir(path):

        category = p.split(".")[0]

        category  = convert(category)

        img_array = cv2.imread(os.path.join(path,p), cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array,dsize=(80,80))

        X.append(new_img_array)

        Y.append(category)

        
create_test_data(path)

X = np.array(X).reshape(-1, 80,80,1)

Y = np.array(Y)
import pickle



pickle.dump( X, open( "train_x", "wb" ) )

pickle.dump( Y, open( "train_y", "wb" ) )
loaded_model = pickle.load(open("train_x", 'rb'))
loaded_model
#Normalize data

X = X/255.0
model = Sequential()

# Adds a densely-connected layer with 64 units to the model:

model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = X.shape[1:]))

model.add(MaxPooling2D(pool_size = (2,2)))

# Add another:

model.add(Conv2D(64,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

# Add another one

model.add(Conv2D(128,(3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())

model.add(Dense(64, activation='relu'))



# Add a softmax layer with 10 output units:

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])



model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)
# preporcessing our test data



train_dir = "test1/test1"

path = os.path.join(main_dir,train_dir)

#os.listdir(path)



X_test = []

id_line = []

def create_test1_data(path):

    for p in os.listdir(path):

        id_line.append(p.split(".")[0])

        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

        new_img_array = cv2.resize(img_array, dsize=(80, 80))

        X_test.append(new_img_array)

create_test1_data(path)

X_test = np.array(X_test).reshape(-1,80,80,1)

X_test = X_test/255
predictions = model.predict(X_test)
predicted_val = [int(round(p[0])) for p in predictions]
print(predicted_val)
submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})
submission_df.to_csv("submission.csv", index=False)