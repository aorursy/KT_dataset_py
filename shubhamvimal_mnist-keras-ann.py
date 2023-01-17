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
train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')

test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
X_train = np.array(train.iloc[:,1:]) # (60000,784)

y_train = np.array(train['label'])   # (60000,)



X_test = np.array(test.iloc[:,1:]) # (10000,784)

y_test = np.array(test['label']) # (10000,)
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,10)

y_test = np_utils.to_categorical(y_test,10)
# Let's make the ANN!



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

# init - Weight....

# uniform - Weights are initialized to small uniformly random values between 0 and 0.05. 

classifier.add(Dense(input_dim = 784, output_dim = 512, init = 'uniform', activation = 'relu'))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))



# Adding the third hidden layer

classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))
# Metrics are evaluated by the model during training.

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 100)