# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import cv2 as cv

from tqdm import tqdm



files = []

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if '/kaggle/input/manwomandetection/dataset/dataset/' not in os.path.join(dirname, filename):

            files.append(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(files)
X = []

y = []



for file in tqdm(files):

    img = cv.imread(file)

    img = cv.resize(img, (50, 50))

    

    gender = 0

    

    X.append(img)

    

    if file.split('/')[-2]=='woman':

        gender = 1

    

    y.append(gender)
X = np.array(X)

y = np.array(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train/255

X_test = X_test/255
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='accuracy',mode='max',min_delta=0.005,verbose=7,patience=5)
model = Sequential()



model.add(Conv2D(64, 3, activation='relu', input_shape=(50, 50,3) ))



model.add(MaxPooling2D((2, 2)))



model.add(Dropout(0.5))



model.add(Conv2D(32, 3, activation='relu'))



model.add(MaxPooling2D((2, 2)))



model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(128,activation='relu'))



model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train, epochs=1000, callbacks=[early_stop])
predictions = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))