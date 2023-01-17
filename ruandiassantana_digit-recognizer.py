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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
from sklearn.model_selection import train_test_split
X = train.drop('label',axis=1)

y = train['label']
X = X.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
X = X/255

test = test/255
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='accuracy',mode='max',min_delta=0.005,verbose=7,patience=5)
model = Sequential()



model.add(Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1) ))



model.add(MaxPooling2D((2, 2)))



model.add(Dropout(0.5))



model.add(Conv2D(32, 3, activation='relu'))



model.add(MaxPooling2D((2, 2)))



model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(128,activation='relu'))



model.add(Dense(10,activation='softmax'))



model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X,y, epochs=1000, callbacks=[early_stop])
predictions = model.predict_classes(test)
df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
df['Label'] = predictions
df.to_csv('submission.csv',line_terminator='\r\n', index=False)