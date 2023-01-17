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
import numpy as np
import pandas as pd
input=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
input.head()
features=input.iloc[:,1:].values.reshape(((42000,28,28,1)))
labels = input.iloc[:, 0].values.reshape((42000,1))
test = test.values.reshape((28000,28,28,1))
test
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Activation,Conv2D,MaxPooling2D
from keras.activations import relu
X_train, X_dev, Y_train, Y_dev = train_test_split(features,
                                                   labels,
                                                   test_size = 0.025,
                                                   random_state = 0,
                                                   stratify = labels)
X_train=X_train/255
X_dev=X_dev/255
X_train
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

Y_train, Y_dev = convert_to_one_hot(Y_train, 10), convert_to_one_hot(Y_dev, 10)
Y_train
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam",
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.summary()
model.PREDICT(X_train, Y_train, epochs=1, batch_size=32)
score = model.evaluate(X_dev, Y_dev, batch_size=32)
score
pred = model.predict_classes(test)
submissions = pd.DataFrame({'ImageId': np.arange(1 , 1 + test.shape[0]), 'Label': pred.astype(int)})
submissions.to_csv('/kaggle/working/cnn1_submission.csv', index=False)