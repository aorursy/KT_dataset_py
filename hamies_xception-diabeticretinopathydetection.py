# This Python 3 environment comes with many helpful analytics libraries installed



# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import cv2

import glob

import os.path

import numpy as np

import pandas as pd

from PIL import Image

from keras.models import *

from keras.layers import *

import keras.backend as K

from keras.callbacks import *

from keras.optimizers import *

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.regularizers import l2

from keras.preprocessing import image

from keras.utils import to_categorical

from keras.applications.xception import Xception

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



img = mpimg.imread("../input/386_left.jpeg")

plt.imshow(img)
filelist = glob.glob('../input/*.jpeg') 

csv = pd.read_csv("../input/trainLabels.csv")

np.size(filelist)

x = []

y = []

for file in filelist:

    img = cv2.resize( cv2.imread(file, cv2.IMREAD_COLOR), (299, 299))

    x.append(np.array(img))

    f = file

    f = f.replace("../input/","")

    f = f.replace(".jpeg","")

    y.append( csv.loc[ csv.image == f, 'level'].values[0] )

    

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.33, random_state=42)
x_train = np.array(list(x_train))

y_train = np.array(list(y_train))

x_test = np.array(list(x_test))

y_test = np.array(list(y_test))

x_train =x_train // 255

x_test = x_test // 255
csv.head()
Y_train = to_categorical(y_train, 5)

Y_test = to_categorical(y_test, 5)

model =Xception(include_top=True, weights=None, input_tensor=None, input_shape=(299, 299, 3), pooling=None, classes=5)





model.summary()

model.compile(

    loss='binary_crossentropy',

    optimizer=RMSprop(lr=1e-4),

    metrics=['accuracy']

)

model.fit(x_train, Y_train, validation_data=(x_test,Y_test), batch_size = 16, epochs=40

          , shuffle=True, verbose=2)

def predict(test_num):

    pre = model.predict(x_test)[test_num]

    index_ = np.argmax( pre )

    categories = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

    return categories[index_]

    
predict(40)