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
#Reading data

data = pd.read_csv('/kaggle/input/iris/Iris.csv')
#Display first five rows of data

data.head()
#get to know about number of species (classes)

data['Species'].unique()
#Length of data

data.shape
#Column names

data.columns
#if there is any NAN value

data.isnull().values.any()
#seperating X_features

X_data = data.drop(['Id','Species'],axis=1)
X_data.shape
#seperating labels and converting categorial labels to numbers.

Y_data = pd.get_dummies(data["Species"], columns=["Species"])
Y_data.head()
Y_data.shape
#Preprocessing

from sklearn import preprocessing
MinMaxScaler = preprocessing.MinMaxScaler()

X_data_minmax = MinMaxScaler.fit_transform(X_data)
#To DataFrame

X_data_frame = pd.DataFrame(X_data_minmax,columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
X_data_frame.head()
#Train_test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data_frame, Y_data,

                                                    test_size=0.2, random_state = 1)
X_train.shape
#NN based classifier using Keras

from keras.models import Sequential

from keras.layers import Dense

import keras.backend as K
K.clear_session()

model = Sequential()

model.add(Dense(6,input_shape = (None,4), activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=8,

          epochs=200, verbose=1, validation_split=0.1)
model.evaluate(X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier
ytrain_knn = np.argmax(y_train.values, axis = 1)
ytrain_knn
y_train.head()
ytest_knn = np.argmax(y_test.values, axis = 1)
#Selecting different Ks in range [1,10] and choose the one with high accuracy

from sklearn import metrics

K_range = 11

acc = [] #np.zeros((Ks-1))

#ConfustionMx = [];

for i in range(1,K_range):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,ytrain_knn)

    yhat=neigh.predict(X_test)

    #acc[n-1] = metrics.accuracy_score(ytest_knn, yhat)

    acc.append(metrics.accuracy_score(ytest_knn, yhat))

    

    #std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



acc
#We can see that with K = 1,2,3,4,6,7,8 we are getting 100 % accuracy, so we are going to select K=6
knn_model = KNeighborsClassifier(n_neighbors = 6).fit(X_train,ytrain_knn)
ypred=knn_model.predict(X_test)
metrics.accuracy_score(ytest_knn, ypred)