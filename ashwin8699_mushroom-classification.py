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
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
df['class'] = df['class'].apply(lambda x: 1 if x=='e' else 0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
final = pd.get_dummies(df)
X = final.drop('class' , axis = 1)
y = final['class']
x_train , x_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier=Sequential()
classifier.add(Dense(60,activation='relu',input_dim=117))
classifier.add(Dense(6,activation='relu',input_dim=117))
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=10)
confusion_matrix(y_test ,classifier.predict(x_test)>0.5 )