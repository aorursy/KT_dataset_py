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
!pip install opencv-python -q
import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



from keras import layers

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Model



import keras.backend as K

from keras.models import Sequential



import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)



from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
os.listdir("../input/super-ai-image-classification")
os.listdir('../input/super-ai-image-classification/train/train/')
train_df = pd.read_csv("../input/super-ai-image-classification/train/train/train.csv")

train_df.head()
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def prepareImages(data, m, dataset):

    print("Preparing images")

    X_train = np.zeros((m, 200, 200, 3))

    count = 0

    

    for fig in data['id']:

        #load images into images of size 200x200x3

        img = image.load_img("../input/super-ai-image-classification/train/train/"+dataset+"/"+fig, target_size=(200, 200, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_train
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



def prepare_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    # print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # print(onehot_encoded)



    y = onehot_encoded

    # print(y.shape)

    return y, label_encoder
X = prepareImages(train_df, train_df.shape[0], "images")

X /= 255
y, label_encoder = prepare_labels(train_df['category'])
X.shape, y.shape
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Model

import keras.backend as K
model = Sequential()



model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (200, 200, 3)))



model.add(BatchNormalization(axis = 3, name = 'bn0'))

model.add(Activation('relu'))



model.add(MaxPooling2D((2, 2), name='max_pool'))

model.add(Conv2D(128, (3, 3), strides = (1,1), name="conv1"))

model.add(Activation('relu'))

model.add(AveragePooling2D((3, 3), name='avg_pool'))



model.add(Flatten())

model.add(Dense(500, activation="relu", name='rl'))

model.add(Dropout(0.8))

model.add(Dense(y.shape[1], activation='softmax', name='sm'))



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()
#model.fit_generator(X, epochs=20, validation_data=y)

import gc

history = model.fit(X, y, epochs=20)

gc.collect()
plt.plot(history.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
# import pickle
# from keras.models import load_model



# model.save('AiAT_HW3_Model.h5')
test = os.listdir("../input/super-ai-image-classification/val/val/images")

print(len(test))
col = ['id']

test_df = pd.DataFrame(test, columns=col)

test_df['category'] = ''
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def prepareImages(data, m, dataset):

    print("Preparing images")

    X_test = np.zeros((m, 200, 200, 3))

    count = 0

    

    for fig in data['id']:

        #load images into images of size 200x200x3

        img = image.load_img("../input/super-ai-image-classification/val/val/"+dataset+"/"+fig, target_size=(200, 200, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_test[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_test
X = prepareImages(test_df, test_df.shape[0], "images")

X /= 255
X.shape
predictions = model.predict(X, verbose=1)
for i, pred in enumerate(predictions):

    print(i, pred)

    print(np.argmax(pred))

    test_df.loc[i, 'category'] = ' '.join(str(np.argmax(pred)))

    #print(label_encoder.inverse_transform(pred.argsort()[-1:][::-1]))
test_df.head() , test_df.tail()
test_df.info()
# test_df.to_csv('submission.csv', index=False)

test_df.to_csv('val.csv', index=False)
#result = pd.read_csv('submission.csv')

#result#
#  model = load_model('AiAT_HW3_Model.h5')