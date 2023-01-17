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
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.callbacks import EarlyStopping

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
df = pd.read_csv('/kaggle/input/devanagari-character-set/data.csv')
df.size
X = df.iloc[:, :-1]

y = df.iloc[:, -1]
from sklearn.preprocessing import LabelBinarizer

binencoder = LabelBinarizer()

y = binencoder.fit_transform(y)
X_images = X.values.reshape((92000),32,32)

import matplotlib.pyplot as plt

plt.imshow(X_images[0])

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size = 0.2, random_state=90)

X_train = X_train/255

X_test = X_test/255



#change the dimension from 3 to 5

X_train = X_train.reshape(X_train.shape[0], 32,32,1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 32,32,1).astype('float32')
conv_model = Sequential()



#add first Conv layer wit max pooling

conv_model.add(

Conv2D(32,(4,4),

      input_shape = (32,32,1),

      activation = 'relu',

      name= 'firstConv'

      )

)



conv_model.add(

MaxPooling2D(pool_size=(2,2),

            name='FirstPool'

            )

)



#add second CONv and maxpool

conv_model.add(

Conv2D(64,(3,3),

      activation = 'relu',

      name= 'secondConv'

      )

)



conv_model.add(

MaxPooling2D(pool_size=(2,2),

            name='SecondPool'

            )

)



conv_model.add(Dropout(0.2))   #it will prevent overfitting
# Building Dense neural net on outputs of the Conv Net



# Input Layer : Flattening the Outputs of the Conv Nets

conv_model.add(Flatten())



# Two Dense Layers 128 Neuraons and 50 Neurons

conv_model.add(

    Dense(128,

          activation='relu',

          name="dense_1"

         )

)

conv_model.add(

    Dense(50, 

          activation='relu', 

          name="dense_2"

         )

)



# Output Layer with 46 Unique Outputs

conv_model.add(

    Dense(46, 

          activation='softmax', 

          name="modeloutput"

         )

)



conv_model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam',

    metrics=['accuracy']

)
conv_model.summary()
result = conv_model.fit(X_train, y_train, validation_split =0.2, epochs = 10, batch_size = 92, verbose=2)
scores = conv_model.evaluate(X_test, y_test, verbose=0)



print("Accuracy: %.2f%%" %(scores[1]*100))

num = 5555

plt.imshow(X_images[num])

plt.show()





imgTrans = X_images[num].reshape(1,32,32,1)

imgTrans.shape



predictions = conv_model.predict(imgTrans)

binencoder.classes_[np.argmax(predictions)]