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



from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

#Retrieved references:https://www.kaggle.com/uciml/pima-indians-diabetes-database

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

data=loadtxt('../input/pimaindiansdiabetesdata/pima-indians-diabetes.data.csv',delimiter=',')

x=data[:,0:8]

y=data[:,8]



myModel=Sequential()

myModel.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))

myModel.add(Dense(units=4,kernel_initializer='uniform',activation='relu'))

myModel.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

myModel.add(Dense(1,activation='sigmoid'))



myModel.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

myModel.fit(x,y,epochs=100,batch_size=10,verbose=0)

#_,accuracy=myModel.evaluate(x,y)

#print("Accuracy %.2f "%(accuracy*100))

#predictions=myModel.predict(x)

#rounded=[round(x[0]) for x in predictions]

predictions=myModel.predict_classes(x)

for i in range(5):

    print("%s=>%d(expected %d)"% (x[i].tolist(),predictions[i],y[i]))