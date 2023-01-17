# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

X_train=train_data.drop(columns='label').values

y_train=train_data.label.values

if 'label' in test_data.columns:

    print(True)

else:

    print(False)

X_test=test_data
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

X_train=sc_x.fit_transform(X_train)

X_test=sc_x.transform(X_test)

print(X_train.shape, y_train.shape, X_test.shape)
X_traincnn=X_train.reshape(X_train.shape[0],X_train.shape[1],1)

X_testcnn=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

print(X_traincnn.shape,X_testcnn.shape)
from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Activation

model=Sequential()

model.add(Conv1D(128,5,input_shape=(784,1)))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Conv1D(64,5))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(MaxPooling1D(pool_size=(10)))

model.add(Conv1D(32,5))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(MaxPooling1D(pool_size=(10)))

model.add(Flatten())

model.add(Dense(y_train.shape[0]))

model.add(Activation('softmax'))

model.summary()

model.save('/kaggle/working/model.h5')
'''import keras

loaded_model=keras.models.load_model('/kaggle/working/model.h5')

#loaded_model.summary()

X1=loaded_model.layers[-2].output

print(X1)

new1=Dense(y_train.shape[0],activation='softmax')(X1)

print(new1)'''
import keras

opt=keras.optimizers.rmsprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
cnnhistory=model.fit(X_traincnn,y_train,epochs=1000,batch_size=16)