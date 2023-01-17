from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import scale
from keras.utils import plot_model

import pandas as pd
import numpy as np

from matplotlib import cm
from sklearn.model_selection import train_test_split


#import dataset
data = pd.read_csv('../input/data.csv')

#changing 'M' and 'B' labels to 1s and 0s
data['diagnosis'].replace('M', 1.0,inplace=True)
data['diagnosis'].replace('B', 0.0,inplace=True)

#visualizing a couple of features
data.plot.scatter(x='radius_mean', y='concavity_mean', c='diagnosis',colormap = cm.get_cmap('Spectral'))
#changing dataframe to numpy array
npdata = data.values
#splitting train-dev-test
train, test = train_test_split(npdata, test_size=0.3)
dev, test = train_test_split(test, test_size=0.5)

X_train = train[:, 2:32]
Y_train = train[:, 1]
X_dev = dev[:, 2:32]
Y_dev = dev[:, 1]
X_test = test[:, 2:32]
Y_test = test[:, 1]
#preprocessing, z score standardization
mean = np.mean(X_train, axis = 0)
stddev = np.std(X_train, axis = 0)
X_train = (X_train - mean)/stddev
X_dev = (X_dev - mean)/stddev
X_test = (X_test - mean)/stddev
#model setup and train
input_dim = X_train.shape[1]
model = Sequential()
model.add(Dense(1, input_dim=input_dim, activation='sigmoid')) 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, batch_size=1, epochs=64, verbose=2)
#model evaluation
model.evaluate(x=X_dev, y=Y_dev, verbose=1)
cost, accuracy = model.evaluate(x=X_test, y=Y_test, verbose=1)
print('Accuracy:' + str(accuracy))