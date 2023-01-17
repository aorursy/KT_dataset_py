# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_path = os.path.join("../input/", "train.csv")
human_activities = pd.read_csv(train_path)
test_path = os.path.join("../input/", "test.csv")
ha_test = pd.read_csv(test_path)

human_activities.info()
ha_test.info()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD,Adadelta,Adagrad
import keras
X = human_activities.drop("activity", axis=1)
y = human_activities["activity"].copy()
y = pd.get_dummies(y)
y= np.argmax(y.values,axis=1)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, random_state=0)
y_train.shape
model = Sequential()
model.add(Dense(32,activation='sigmoid',input_dim=562))
model.add(Dropout(0.5))
model.add(Dense(6,activation='softmax'))
adadelta = Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(optimizer=adadelta,
              loss='mse',
              metrics=['accuracy'])
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=6)
model.fit(X_train,y_train_one_hot,epochs=2000,batch_size=64)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=6)
model.evaluate(X_test,y_test_one_hot)
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
# y_pred
print(confusion_matrix(y_test, np.argmax(y_pred,axis=1)))
from sklearn.metrics import classification_report
print(classification_report(y_test,  np.argmax(y_pred,axis=1), target_names=["LAYING", "SITTNG", "STANDNG", "WALKING", "WALKING_DOWNSTAIRS", "WALKING_UPSTAIRS"]))
