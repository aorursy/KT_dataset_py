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
import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix



from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
train_path = '../input/digit-recognizer/train.csv'

test_path = '../input/digit-recognizer/test.csv'

submission_path = '../input/digit-recognizer/sample_submission.csv'
train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

submission = pd.read_csv(submission_path)
x_train = train.drop(['label'], axis=1)

Y_train = train['label']
#normalizing

X_train = x_train/255.0

X_test = test/255.0
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
x_train = x_train.values.reshape(-1, 28, 28, 1)

x_test = x_test.values.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
#tensorflow sequential model

model = Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', input_shape=(28,28,1)))

model.add(layers.Activation('relu'))

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same'))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding='same'))

model.add(layers.Activation('relu'))

model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding='same'))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))



model.add(layers.Dropout(0.25))

model.add(layers.Flatten())



model.add(layers.Dense(units=256, activation='relu'))

model.add(layers.Dense(units=10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model_chart.png', show_shapes=True)
#data augmentation



datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

                    samplewise_center=False,  # set each sample mean to 0

                    featurewise_std_normalization=False,  # divide inputs by std of the dataset

                    samplewise_std_normalization=False,  # divide each input by its std

                    zca_whitening=False,  # apply ZCA whitening

                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

                    zoom_range = 0.1, # Randomly zoom image 

                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

                    horizontal_flip=False,  # randomly flip images

                    vertical_flip=False)



datagen.fit(x_train)
#fitting

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),

                           epochs=7,

                           validation_data=(x_test, y_test),

                           steps_per_epoch=len(x_train)//64)
y_pred = model.predict_classes(x_test)
y_test = y_test.to_list()

y_test = np.array(y_test)
print('accuracy score: ',accuracy_score(y_test, y_pred))
print('confusion matrix: ')

print(confusion_matrix(y_test, y_pred))
X_test = X_test.values.reshape(-1, 28, 28, 1)

y_submission_pred = model.predict_classes(X_test)
y_preds = pd.Series(y_submission_pred)

image_id = pd.Series(np.arange(1,28001))
data = {'ImageId':image_id,

        'Label':y_preds}



final = pd.DataFrame(data)
final.to_csv('submission.csv', index=False)