# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
mushroom_df = pd.read_csv('../input/mushrooms.csv')
mushroom_df.head()
#Target varaible : class
Y = mushroom_df['class']
X = mushroom_df.drop(['class'],axis=1)


Y = pd.get_dummies(Y)
X = pd.get_dummies(X)
print(Y.shape)
print(X.shape)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3)
keras_model = Sequential()
keras_model.add(Dense(15,input_dim = x_train.shape[1],activation='relu'))
keras_model.add(Dense(y_train.shape[1],activation='softmax'))
keras_model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
# Fit the model
model = keras_model.fit(x_train,y_train,epochs=10)
#Save the model
keras_model.save('mushroom_model.h5')
pred_model = load_model('mushroom_model.h5')
y_pred = pred_model.predict(x_test)
print('Loss : ',mean_squared_error(y_test.values,y_pred.round()))
