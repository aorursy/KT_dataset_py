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
dataset = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

dataset_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

y = dataset.iloc[:,0].values.astype('int32')

X = dataset.iloc[:,1:].values.astype('float32')

test_X = dataset_test.values.astype('float32')
# import matplotlib.pyplot as plt

# #Convert train datset to (num_images, img_rows, img_cols) format 

# X = X.reshape(X.shape[0], 28, 28)



# for i in range(6, 9):

#     plt.subplot(330 + (i+1))

#     plt.imshow(X[i], cmap=plt.get_cmap('gray'))

#     plt.title(y[i]);
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 40)
import keras 

from keras.models import Sequential

from keras.layers import Dense
from keras.utils import to_categorical

# X_train = to_categorical(X_train)

y_train = to_categorical(y_train)

#y_test = to_categorical(y_test)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout

from keras.activations import relu, sigmoid

from keras.layers import LeakyReLU
# def create_model(layers, activation):

#     model = Sequential()

#     for i, nodes in enumerate(layers):

#         if i==0:

#             model.add(Dense(nodes,input_dim=X_train.shape[1]))

#             model.add(Activation("relu"))

#         elif i==enumerate(layers):

#             model.add(Dense(nodes,input_dim=X_train.shape[1]))

#             model.add(Activation("softmax"))

#         else:

#             model.add(Dense(nodes))

#             model.add(Activation("relu"))

#     model.add(Dense(1)) # Note: no activation beyond this point

    

#     model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#     return model

    

# model = KerasClassifier(build_fn=create_model, verbose=0)
# layers = [[397, 397], [784,784, 10], [397, 397, 10],[784,784,784, 10], [397, 397, 397, 10]]

# activations = ['relu','softmax']

# param_grid = dict(layers=layers, activation=activations, batch_size = [50, 100, 128, 256], epochs=[30, 100])

# grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=50)
# grid_result = grid.fit(X_train, y_train)
# [grid_result.best_score_,grid_result.best_params_]
#Initialising the ANN

classifier = Sequential()

#adding 1st layer

classifier.add(Dense(output_dim = 784, init = "uniform", activation = "relu", input_dim = 784))

#adding more layer

classifier.add(Dense(output_dim = 397, init = "uniform", activation = "relu"))

classifier.add(Dense(output_dim = 397, init = "uniform", activation = "relu"))

classifier.add(Dense(output_dim = 397, init = "uniform", activation = "relu"))

classifier.add(Dense(output_dim = 397, init = "uniform", activation = "relu"))

#adding final layer

classifier.add(Dense(output_dim = 10, init = "uniform", activation = "softmax"))

#Compiling ANN

classifier.compile(optimizer = "adam", loss = "categorical_crossentropy" )

#fitting

classifier.fit(X_train, y_train, batch_size = 150, nb_epoch = 100)
y_pred = classifier.predict(X_test)

y_pred = np.argmax(y_pred,axis = 1)

y_pred = pd.Series(y_pred,name="Label")
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
y_pred = classifier.predict(test_X)

y_pred = np.argmax(y_pred,axis = 1)

y_pred = pd.Series(y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred],axis = 1)



submission.to_csv('sample_submission1.csv', index=False)