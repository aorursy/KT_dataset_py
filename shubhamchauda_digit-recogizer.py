# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tensorflow import keras





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_data.head()

disp = train_data.iloc[2].values[1:]

disp = disp.reshape(28,28)

plt.imshow(disp)
y_train = (train_data['label']).to_numpy()

X_train = (((train_data.copy()).drop('label',axis = 1))/255)

X_train = (X_train.to_numpy()).reshape((len(train_data),28,28))

X_test = (test_data.copy())/255

X_test = (X_test.to_numpy()).reshape((len(test_data),28,28))

X_test.shape

model = keras.Sequential([

             #keras.layers.Conv2D(filters = 32,

             #                   activation = 'relu',

             #                   kernel_size = 4,

             #                   input_shape = (28,28,1)),

             #keras.layers.Flatten(),

        

             #keras.layers.Dense(units =112,

             #                  activation = 'relu'

             #               ),

             #keras.layers.Dense(10,activation= 'softmax')

         

        keras.layers.Flatten(input_shape =(28,28)),

        keras.layers.Dense(512, activation='relu'),

        keras.layers.Dense(128, activation='relu'),

        keras.layers.Dense(10,   activation='softmax')

                             ])

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.01),loss ='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10, validation_split=0.3)
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)

submission = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})

submission.to_csv('submission.csv', index=False)

disp = X_test[50].reshape(28,28)

plt.imshow(disp)

y_pred[50]