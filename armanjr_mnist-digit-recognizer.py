# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tensorflow import keras



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



file = "/kaggle/input/digit-recognizer/train.csv"

data = np.loadtxt(file, skiprows=1, delimiter=',')

x, y = prep_data(data)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout



model = Sequential()

model.add(Conv2D(12, kernel_size=(3,3), activation='relu', input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, batch_size=100, epochs=25, validation_split=0.2)
score = model.evaluate(x, y, verbose=0)

print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
def prep_test_data(raw):

    x = raw[:,0:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x



val_file = "/kaggle/input/digit-recognizer/test.csv"

val_data = np.loadtxt(val_file, skiprows=1, delimiter=',')

x_test = prep_test_data(val_data)
predictions = model.predict_classes(x_test)



indexes = [i for i in range(1,len(val_data)+1)]

output = pd.DataFrame({'ImageId': indexes,'Label': predictions})

output.to_csv('submission.csv', index=False)