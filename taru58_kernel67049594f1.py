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
mnist=pd.read_csv("../input/train.csv")
mnist.columns
X_train=mnist.iloc[:,1:].values
y_train=mnist.iloc[:,0].values
X_train
X_train=X_train/255.0
X_train.shape
y_train.shape
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier=Sequential()

classifier.add(Dense(output_dim=128,init='uniform',activation='relu',input_dim=784))
classifier.add(Dense(output_dim=64,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=10,init='uniform',activation='softmax'))
classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,epochs=10,batch_size=50)
mnist_test=pd.read_csv("../input/test.csv")
X_test=mnist_test.values
X_test
X_test=X_test/255.0
pred=classifier.predict(X_test)
type(pred)
pred=np.argmax(pred,axis=1)
pred.shape
mnist_sub=pd.read_csv('../input/sample_submission.csv')
mnist_sub.head()
mnist_sub['Label']=pred
mnist_sub.to_csv('sample_submission.csv',index=False)