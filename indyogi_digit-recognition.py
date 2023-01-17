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
import keras.models as models 

import keras.layers as layers

from keras.utils import to_categorical



# load the training and test datasets 



train_data = pd.read_csv ('../input/digit-recognizer/train.csv')

test_data = pd.read_csv ('../input/digit-recognizer/test.csv')



# the labels in the training set are in the first (0th column). Let us first extract the labels.

train_labels = train_data['label']



# drop the first column from the test set 

train_data.drop ('label', axis = 1, inplace = True)

train_data = train_data.astype ('float') / 255



# view the sample inputs from the training dataset

#train_data.head()
diginet = models.Sequential()

diginet.add (layers.Dense (512, activation = 'relu', input_shape = (784,)))

diginet.add (layers.Dense (128, activation = 'relu'))

diginet.add (layers.Dense (10, activation = 'softmax'))

diginet.summary()

diginet.compile (optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#print ('train labels shape is',train_labels.shape)

#print (train_labels[0:5])

train_labels_onehot = to_categorical(train_labels)

#print (train_labels_onehot[0:5])
# train the model 

diginet.fit (train_data, train_labels_onehot, epochs = 15, batch_size = 128)
pred = diginet.predict (test_data)


# create the sample submission file 

opfile = open ('submission.csv', 'a')

opfile.write ("ImageId,Label\n")



# create an array of size (#test_data, 2)

predarray = np.zeros((test_data.shape[0], 2))



rows = [''] * pred.shape[0]



for i in range (test_data.shape[0]):

    rows[i] = '%d,%d\n' % (i+1,np.argmax(pred[i]))

    

opfile.writelines (rows) 

opfile.close()