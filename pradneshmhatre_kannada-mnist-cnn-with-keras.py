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
from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Flatten,Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import RMSprop,Adam, SGD

CV = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

x_cv = CV.loc[:,'pixel0':].values

y_cv = CV.loc[:,:'label'].values

x_train = train.loc[:,'pixel0':].values

y_train = train.loc[:,:'label'].values
print('x_train shape: {} y_train shape: {}'.format(x_train.shape, y_train.shape))

print('x_cv shape: {} y_cv shape: {}'.format(x_cv.shape, y_cv.shape))
y_train = to_categorical(y_train)

y_cv = to_categorical(y_cv)
num_classes = y_train.shape[1]
img_rows = 28

img_cols = 28



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_cv = x_cv.reshape(x_cv.shape[0], img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_cv = x_cv.astype('float32')



x_train = x_train/255

x_cv = x_cv/255



print('x_train shape: {} y_train shape: {}'.format(x_train.shape, y_train.shape))

print('x_cv shape: {} y_cv shape: {}'.format(x_cv.shape, y_cv.shape))
model = Sequential()



model.add(Conv2D(32, kernel_size=(5,5), activation= 'relu', padding = 'Same',input_shape=input_shape))

model.add(Conv2D(64, (3,3), activation= 'relu'))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3,3), activation= 'relu', padding = 'Same'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation= 'relu'))

model.add(Dropout(0.50))

model.add(Dense(num_classes, activation= 'softmax'))



model.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
batch_size = 32

epochs = 20



history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_cv, y_cv))
import matplotlib.pyplot as plt



history_dict = history.history



tr_loss = history_dict['loss']

cv_loss = history_dict['val_loss']

epochs = range(1, len(tr_loss)+1)

plt.plot(epochs,tr_loss, label='Train Loss')

plt.plot(epochs,cv_loss, label='CV Loss')

plt.title('Loss Evaluation')

plt.grid()

plt.legend()

plt.show()
history_dict = history.history



tr_acc = history_dict['accuracy']

cv_acc = history_dict['val_accuracy']

epochs = range(1, len(tr_acc)+1)

plt.plot(epochs,tr_acc, label='Train Acc')

plt.plot(epochs,cv_acc, label='CV Acc')

plt.title('Accuracy Evaluation')

plt.grid()

plt.legend()

plt.show()
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

test_id = test.loc[:,:'id'].values

x_test = test.loc[:,'pixel0':].values

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_test = x_test.astype('float32')

x_test = x_test/255
pred = model.predict(x_test)
label = np.argmax(pred, axis=1)
test_csv = pd.DataFrame(test['id'].values,columns=['id'])
test_csv['label'] = label
test_csv.to_csv('submission.csv', index=False)