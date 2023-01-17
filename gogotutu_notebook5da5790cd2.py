# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

import numpy as np

import keras.utils.np_utils as kutils



# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv").values

test  = pd.read_csv("../input/test.csv").values



# Write to the log:

print ("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print ("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs
trainY = train[:,0]

trainX = train[:,1:]

trainY = kutils.to_categorical(trainY)
# create model

model = Sequential()

model.add(Dense(20, input_dim=784, init='uniform', activation='relu'))

model.add(Dense(20, init='uniform', activation='relu'))

model.add(Dense(20, init='uniform', activation='relu'))

model.add(Dense(10, init='uniform', activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.fit(trainX, trainY, nb_epoch=1, batch_size=10)
# evaluate the model

scores = model.evaluate(trainX, trainY)

print 

print ("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
Ytest = model.predict(test)

YY = np.zeros(Ytest.shape[0])



for k in range(Ytest.shape[0]):

    YY[k] = np.argmax(Ytest[k])

np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(YY)+1),YY], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')