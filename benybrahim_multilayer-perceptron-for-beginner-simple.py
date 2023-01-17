import numpy as np # Array manipulation

import pandas as pd # Dataframe manipulation



# Multilayer perceptron Neural Network

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import np_utils
# Load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
# Extract images pixels

images = train.iloc[:,1:].values

images = images.astype(np.float)



# Extract numbers Labels

labels = train.iloc[:,0].values
# Normalize input from 0-255 to 0-1

images = images / 255.0

num_pixels =  images.shape[1]



# one hot encode outputs

labels = np_utils.to_categorical(labels)

num_classes = labels.shape[1]
# define baseline model

def mlp_model():

	# create model

	model = Sequential()

	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))

	model.add(Dense(num_classes, init='normal', activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
# build the model

model = mlp_model()

# Fit the model

model.fit(images, labels, nb_epoch=10, batch_size=200, verbose=2)
# use the NN model to classify test data

pred = model.predict(test.values)

pred = pred.argmax(1) # transform the binary matrix to an array of 0 to 9 labels

# save results

np.savetxt('submission_MLP.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')