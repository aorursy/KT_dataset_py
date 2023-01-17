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

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from sklearn.metrics import confusion_matrix

import numpy as np
classifier = Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(216,activation="relu"))

classifier.add(Dense(1,activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1/255)

test_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train',target_size = (64,64),batch_size = 1,class_mode = 'binary',shuffle=False) 

test_set = test_datagen.flow_from_directory('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test',target_size = (64,64),batch_size = 1,class_mode = 'binary',shuffle=False) 

Hist = classifier.fit_generator(training_set,steps_per_epoch = 80,epochs = 20,validation_data = test_set,validation_steps = 20)

eval_result = classifier.evaluate_generator(test_set)

print('Loss Rate at evaluation data :', eval_result[0])

print('Accuracy Rate at evaluation data :', eval_result[1])
y_pred = classifier.predict_generator(test_set)
y_pred
y_pred = np.round(y_pred)

y_pred
y_test = test_set.classes
print(confusion_matrix(y_test,y_pred))