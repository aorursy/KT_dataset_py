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
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Flatten
from keras.preprocessing.image import ImageDataGenerator
train_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test_data = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
X_train = train_data.iloc[:,1:785].values.reshape(-1,28,28,1)
Y_train = train_data.iloc[:,0].values.reshape(27455,1)
X_test = test_data.iloc[:,1:785].values.reshape(-1,28,28,1)
Y_test = test_data.iloc[:,0].values.reshape(7172,1)
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
Y_train = label_binarizer.fit_transform(Y_train)
Y_test = label_binarizer.fit_transform(Y_test)

X_train = X_train/255
X_test = X_test/255

gen = ImageDataGenerator( rotation_range = 10, zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1  )
gen.fit(X_train)
model = Sequential()

model.add(Conv2D(70 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(40 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(20 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(5, (3,3), strides = 1, padding = 'same', activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2), strides = 2, padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 50 , activation = 'relu'))

model.add(Dense(units = 24 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()


history = model.fit(gen.flow(X_train,Y_train, batch_size = 512) ,epochs = 50 , validation_data = (X_test, Y_test))

print("Accuracy of the model is : " , model.evaluate(X_test,Y_test)[1]*100 , "%")

