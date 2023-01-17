# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop


import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv( '../input/train.csv')
test = pd.read_csv( '../input/test.csv' )
train_matrix = train.as_matrix()
x_train = train_matrix[:,1:]
y_train = train_matrix[:,0]

x_train = x_train.reshape( ( 42000, 28, 28, 1 ) ).astype( 'float32' ) / 256.0
x_val = x_train[ 40000:, :, :, :]
x_train = x_train[ :40000, :, :, :]
x_test = test.as_matrix().reshape( 28000, 28,28,1).astype( 'float32' ) / 256.0

y_train = to_categorical( y_train, 10 )
y_val = y_train[40000:]
y_train = y_train[:40000]
x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape


image_count = 10
images = [ x.reshape( 28,28 ) for x in np.split(x_train[:image_count,:,:,0], image_count ) ]
plt.imshow( np.hstack( images ) )
simple = Sequential()
simple.add( Flatten( input_shape=(28,28,1)))
simple.add( Dense( 512, activation='relu' ))
simple.add( Dense( 10, activation='softmax' ))

simple.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
simple.summary()
simple.fit( x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val,y_val))
cnn = Sequential()

cnn.add( Conv2D( 16, 3, input_shape =( 28,28, 1)))
cnn.add( MaxPooling2D())
cnn.add( Conv2D( 32, 3, ))
cnn.add( Flatten())
cnn.add( Dense( 128, activation='relu' ))
cnn.add( Dense( 128, activation='relu' ))
cnn.add( Dense( 10, activation='softmax' ))

cnn.compile( loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
cnn.summary()
cnn.fit( x_train, y_train, batch_size=128, epochs=5, validation_data=(x_val,y_val))
                    
results = np.argmax( cnn.predict( x_test ), axis=1)

output = pd.DataFrame( {'Label': results, 'ImageId' : range( 1, len( results ) + 1) } )
output
plt.hist( output)
output.to_csv( 'results.csv', index=False)
