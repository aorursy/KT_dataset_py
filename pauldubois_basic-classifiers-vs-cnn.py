# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import pandas as pd 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

validation = pd.read_csv('/kaggle/input/digit-recognizer/test.csv') 
print('data shape: ', data.shape)

print('validation shape', validation.shape)
data.head()
validation.head()
from sklearn.utils import resample
data_sample = resample(data, n_samples = 5000)
data_sample.shape
import matplotlib.pyplot as plt
X = data_sample.drop(['label'], axis = 1)

y = data_sample.label
plt.hist(y)

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
from sklearn import svm

svc = svm.SVC()

svc.fit(X_train, y_train)

svc.score(X_test, y_test)
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    raw = raw.to_numpy()

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



X, y = prep_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
#define

model = Sequential()



model.add(Conv2D(12, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))



model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(60, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



#compile

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



#fit

model.fit(X_train, y_train,

          validation_data = (X_test, y_test),

          batch_size = 32,

          epochs = 10)

#note: you have multiple ways of doing the validation. The validation_data (and thus the train/test split) is not necessary

#as you can use the validation_split argument that will automatically select a part of the input data for the validation
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy : %.2f%%" % (scores[1]*100))
model.save("digitRecognizerModel.h5")

print("The model has been saved!")
preds = model.predict_classes(X_test)
#our y_test was categorical, so let's get it back to normal

y_test_transfo = y_test.argmax(1)
wrong_preds = []

for i in range(0, len(preds), 1):

    if preds[i].any() != y_test_transfo[i].any():

        wrong_preds.append(i)
plt.figure(figsize=(15,25))

i=1

for j in wrong_preds:

    plt.subplot(10,5,i)

    plt.axis('off')

    plt.imshow(X_test[j][:,:,0])

    pred_classe = preds[j].argmax(axis=-1)

    plt.title('Prediction : %d / Label : %d' % (preds[j], y_test_transfo[j]))

    i+=1
validation = validation.to_numpy()



num_images = validation.shape[0]

x_val = validation.reshape(num_images, img_rows, img_cols, 1)

x_val = x_val / 255



y_val = model.predict_classes(x_val)
sub = pd.DataFrame()

sub['ImageId'] = list(range(1, validation.shape[0] + 1))

sub['Label'] = y_val
sub.to_csv('sub_file.csv', index=False)