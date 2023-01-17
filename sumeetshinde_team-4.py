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
import os

from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials
import zipfile

from google.colab import drive



drive.mount('/content/drive/')
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import regularizers
# below is a model architecture that might work, I'm very open to editing it for better performance.

model = Sequential() 



model.add(Conv2D(32, (9, 9), input_shape=all_images.shape[1:])) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 



model.add(Conv2D(32, (7, 7), input_shape=all_images.shape[1:])) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

  

model.add(Conv2D(32, (5, 5))) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 

  

model.add(Conv2D(32, (3, 3))) 

model.add(Activation('relu')) 

model.add(MaxPooling2D(pool_size=(2, 2))) 





  

model.add(Flatten()) 



model.add(Dense(32)) 

model.add(Activation('relu')) 





model.add(Dense(16))

model.add(Activation('relu'))



"""

model.add(Dense(8))

model.add(Activation('relu'))

model.add(Dropout(0.25))

"""



model.add(Dense(1))

model.add(Activation('linear'))

# model.add(Dropout(0.25)) 

# model.add(regularizers.l2(0.0
model.compile(loss='mean_squared_error', optimizer='adam') # lets the python interpreter we are done defining the model and it compiles it with an optimizer and loss metric.
EP = 20 # number of epochs

BS = 160 # batch size
model.fit(all_images,train.Distance.values, batch_size=BS, epochs=EP, validation_split=0.1, shuffle=True)

          # shuffle=True)   #Most of these are place holders until we can load in the data, I have also set shuffle to true to make the training independent of the pictures orders.



# classifier.fit(images,train.Category.values, epochs=100, batch_size=32,validation_split=0.1)
test = pd.read_csv('/content/drive/My Drive/sample.csv', error_bad_lines=False)

newid1 = [str(i) for i in test['Id']]



file = os.listdir()



# Loading the training Images

testing = [imread('/content/drive/My Drive/TestingImages.zip/TestingImages/' + j) for j in newid1]

resized1 = [resize(i, (width, height)) for i in testing]

test_images = np.array(resized1)
predictions = model.predict(test_images)
pred = pd.DataFrame(predictions, columns = ['Distance'])
test.Distance=pred
test.to_csv('predictions.csv')