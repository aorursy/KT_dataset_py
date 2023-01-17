import sklearn
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







import os,cv2

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from pylab import rcParams

rcParams['figure.figsize'] = 20, 10



#from sklearn.utils import shuffle

#from sklearn.cross_validation import train_test_split



import keras



from keras.utils import np_utils



from keras import backend as K



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD,RMSprop,adam

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
# get the data

filname = '../input/fer2013/fer2013.csv'

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names=['emotion','pixels','usage']

df=pd.read_csv('../input/fer2013/fer2013.csv',names=names, na_filter=False)

im=df['pixels']

df.head()
a = df['emotion']

idx = pd.Index(a)

count = idx.value_counts()

print(count)
print(idx)
def getData(filname):

    # images are 48x48

    # N = 35887

    Y = []

    X = []

    first = True

    for line in open(filname):

        #This condition skips the first condition

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    #X, Y = np.array(X) / 255.0, np.array(Y)

    X, Y = np.array(X)/255.0 , np.array(Y)

    return X, Y

X, Y = getData(filname)

num_class = len(set(Y))

print(num_class)

N,D = X.shape

X = X.reshape(N, 48, 48, 1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
img = X_train[780].reshape(48,48)

plt.imshow(img, interpolation='nearest')

plt.show()
aug = ImageDataGenerator(

    rotation_range=25, width_shift_range=0.1,

    height_shift_range=0.1, shear_range=0.2, 

    zoom_range=0.2,horizontal_flip=True, 

    fill_mode="nearest")
no_images = 0

no_sadness = 0

no_anger = 0

no_happy = 0

no_fear = 0

no_surprise = 0

no_neutral = 0

#epoch = 26

epoch = 51



for e in range(epoch):

    print('Epoch', e)

    batches = 0

    for x_batch, y_batch in aug.flow(X_train, y_train, batch_size=32):

        batches += 1

        no_images +=len(x_batch)

        y_batch_real = y_batch

        for label in y_batch_real:

            if(label== 4):

                no_sadness+=1

            elif(label==0):

                no_anger+=1

            elif(label==3):

                no_happy+=1

            elif(label==2):

                no_fear+=1

            elif(label==5):

                no_surprise+=1

            elif(label==6):

                no_neutral+=1

        if(e==5):

            emotion_count_5={'sadness':no_sadness,'anger':no_anger,'happy':no_happy,'fear':no_fear,'surprise':no_surprise,'neutral':no_neutral}

        elif(e==15):

            emotion_count_15={'sadness':no_sadness,'anger':no_anger,'happy':no_happy,'fear':no_fear,'surprise':no_surprise,'neutral':no_neutral}

        elif(e==25):

            emotion_count_25={'sadness':no_sadness,'anger':no_anger,'happy':no_happy,'fear':no_fear,'surprise':no_surprise,'neutral':no_neutral}

        elif(e==35):

            emotion_count_35={'sadness':no_sadness,'anger':no_anger,'happy':no_happy,'fear':no_fear,'surprise':no_surprise,'neutral':no_neutral}

        elif(e==50):

            emotion_count_50={'sadness':no_sadness,'anger':no_anger,'happy':no_happy,'fear':no_fear,'surprise':no_surprise,'neutral':no_neutral}

          

        if batches >= len(X_train) / 32:

            # we need to break the loop by hand because

            # the generator loops indefinitely

            break
print(emotion_count_5)
print(emotion_count_15)
print(emotion_count_25)
print(emotion_count_35)
print(emotion_count_50)