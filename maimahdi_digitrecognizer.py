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
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.preprocessing.image import ImageDataGenerator
digit_train = pd.read_csv('../input/digit-recognizer/train.csv')

digit_test = pd.read_csv('../input/digit-recognizer/test.csv')

digit_train.isnull().any().value_counts()
digit_train
y = digit_train["label"]



X= digit_train.drop(labels = "label", axis=1)
X
X = X/255.0

digit_test = digit_test/255.0



X = X.values.reshape(-1, 28, 28, 1)

digit_test = digit_test.values.reshape(-1, 28,28,1)
y = to_categorical(y, num_classes=10)

#Here we split the data into train and test datasets

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (7,7),padding = 'Same',activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (7,7),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(3,3)))

model.add(Dropout(0.50))





model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(3,3), strides=(3,3)))

model.add(Dropout(0.50))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()
model.compile(loss = "categorical_crossentropy",optimizer = Adam(0.0015) , metrics=["accuracy"])

hist = model.fit(x_train, y_train,

                batch_size=86,

                epochs=20,

                validation_data=(x_test,y_test))
final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)

print("loss: {0:.3f},accuracy: {1:.3f}".format(final_loss, final_acc))
preds = model.predict(digit_test)   

preds
pred_labels = model.predict_classes(digit_test)

pred_labels
submission = np.argmax(preds,axis=1)                               # since results are in categorical form, choose the highest in each and store them 

submission  
output = pd.DataFrame({'ImageId': range(1,len(digit_test)+1) ,'Label':submission })               # make dataframe with the column headders and predicted values



output.to_csv("results.csv",index=False)