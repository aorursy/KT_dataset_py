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
import matplotlib.pyplot as plt
dataset = pd.read_excel('/kaggle/input/climate-model-simulation-crashes-data-set/Book1.xlsx')

dataset.head()
X = dataset.iloc[:, 2:20].values 
y=dataset.outcome.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.25, random_state = 24)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier  = Sequential()
classifier.add(Dense(output_dim = 9,init = 'uniform' , activation = 'relu' , input_dim = 18 ))
classifier.add(Dense(output_dim = 5,init = 'uniform' , activation = 'relu' ))
classifier.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist=classifier.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
classifier.evaluate(X_test,y_test)
sig  = Sequential()
sig.add(Dense(output_dim = 9,init = 'uniform' , activation = 'sigmoid' , input_dim = 18 ))
sig.add(Dense(output_dim = 5,init = 'uniform' , activation = 'sigmoid' ))
sig.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
sig.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist1=sig.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
sig.evaluate(X_test,y_test)
sof  = Sequential()
sof.add(Dense(output_dim = 9,init = 'uniform' , activation = 'softmax' , input_dim = 18 ))
sof.add(Dense(output_dim = 5,init = 'uniform' , activation = 'softmax' ))
sof.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
sof.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist2=sof.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100);
sig.evaluate(X_test,y_test);
plt.plot(hist.history['loss'])
plt.plot(hist1.history['loss'])
plt.plot(hist2.history['loss'])
plt.title('Model loss ')
plt.xlabel('Epoch')
plt.legend(['relu', 'sigmoid','softmax'], loc='upper right')
plt.show()
plt.plot(hist.history['accuracy'])
plt.plot(hist1.history['accuracy'])
plt.plot(hist2.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.legend(['relu', 'sigmoid','softmax'], loc='lower right')
plt.show()
classifier1  = Sequential()
classifier1.add(Dense(output_dim = 9,init = 'uniform' , activation = 'relu' , input_dim = 18 ))
classifier1.add(Dense(output_dim = 5,init = 'uniform' , activation = 'relu' ))
classifier1.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
classifier1.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist3=classifier1.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
classifier1.evaluate(X_test,y_test)
sig1  = Sequential()
sig1.add(Dense(output_dim = 9,init = 'uniform' , activation = 'relu' , input_dim = 18 ))
sig1.add(Dense(output_dim = 5,init = 'uniform' , activation = 'relu' ))
sig1.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
sig1.compile(optimizer = 'adamax' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist4=sig1.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
sig1.evaluate(X_test,y_test)
sof1  = Sequential()
sof1.add(Dense(output_dim = 9,init = 'uniform' , activation = 'relu' , input_dim = 18 ))
sof1.add(Dense(output_dim = 5,init = 'uniform' , activation = 'relu' ))
sof1.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
sof1.compile(optimizer = 'sgd' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist5=sof1.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
sig1.evaluate(X_test,y_test)
plt.plot(hist3.history['loss'])
plt.plot(hist4.history['loss'])
plt.plot(hist5.history['loss'])
plt.title('Model loss ')
plt.xlabel('Epoch')
plt.legend(['adam', 'adamax','sgd'], loc='upper right')
plt.show()
plt.plot(hist3.history['accuracy'])
plt.plot(hist4.history['accuracy'])
plt.plot(hist5.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.legend(['adam', 'adamax','sgd'], loc='lower right')
plt.show()
classifier2  = Sequential()
classifier2.add(Dense(output_dim = 9,init = 'uniform' , activation = 'sigmoid' , input_dim = 18 ))
classifier2.add(Dense(output_dim = 5,init = 'uniform' , activation = 'sigmoid' ))
classifier2.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
classifier2.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
hist6=classifier2.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
classifier2.evaluate(X_test,y_test)
sig2  = Sequential()
sig2.add(Dense(output_dim = 9,init = 'uniform' , activation = 'sigmoid' , input_dim = 18 ))
sig2.add(Dense(output_dim = 5,init = 'uniform' , activation = 'sigmoid' ))
sig2.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
sig2.compile(optimizer = 'adam' , loss = 'hinge' , metrics = ['accuracy'] )
hist7=sig2.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
sig2.evaluate(X_test,y_test)
sof2  = Sequential()
sof2.add(Dense(output_dim = 9,init = 'uniform' , activation = 'sigmoid' , input_dim = 18 ))
sof2.add(Dense(output_dim = 5,init = 'uniform' , activation = 'sigmoid' ))
sof2.add(Dense(output_dim = 1,init = 'uniform' , activation = 'sigmoid' ))
sof2.compile(optimizer = 'adam' , loss = 'squared_hinge' , metrics = ['accuracy'] )
hist8=sof2.fit(X_train, y_train, batch_size = 10 , nb_epoch = 100)
sig2.evaluate(X_test,y_test)
plt.plot(hist6.history['loss'])
plt.plot(hist7.history['loss'])
plt.plot(hist8.history['loss'])
plt.title('Model loss ')
plt.xlabel('Epoch')
plt.legend(['binary_crossentropy', 'hinge','squared_hinge'], loc='upper right')
plt.show()
plt.plot(hist6.history['accuracy'])
plt.plot(hist7.history['accuracy'])
plt.plot(hist8.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.legend(['binary_crossentropy', 'hinge','squared_hinge'], loc='lower right')
plt.show()