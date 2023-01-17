# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense , Dropout

from keras.optimizers import RMSprop





train = pd.read_csv("../input/train.csv")

test_images = (pd.read_csv("../input/test.csv").values).astype('float32')

train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')

train_images = train_images / 255

test_images = test_images / 255





from keras.utils.np_utils import to_categorical

train_labels = to_categorical(train_labels)

num_classes = train_labels.shape[1]



seed = 43

np.random.seed(seed)



model = Sequential()

model.add(Dense(64, activation='relu',input_dim=(28 * 28)))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy',

 metrics=['accuracy'])



model.fit(train_images, train_labels, 

	        nb_epoch=1, batch_size=64,verbose=1)

predictions = model.predict_classes(test_images, verbose=1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

	                         "Label": predictions})

submissions.to_csv("DR"+str(i)+".csv", index=False, header=True)