!git clone https://github.com/time-series-tools/Activity2Vec
!conda install -c maxibor tsne -y
! pip install "more-itertools==7.1.0"

# Import packages

import os

import sys

sys.path.insert(0, os.path.abspath('./Activity2Vec/'))

import tensorflow as tf

import numpy as np

from datasets.har import load_data

from model.act2vec import Act2Vec

from util.utils import plot_latent_space

from util.utils import rf

from util.utils import plot_confusion_matrix

from util.utils import print_result
# allow tenserflow to use GPU

physical_devices = tf.config.experimental.list_physical_devices('GPU')

assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['AUTOGRAPH_VERBOSITY'] = '10'

tf.autograph.set_verbosity(0)
n_epochs = 600
(x_train, y_train), (x_test, y_test) = load_data()
print('x_train shape is: ', x_train.shape)

print('x_test shape is: ', x_test.shape)

print('Number of classes: ', len(np.unique(y_train)))
act2vec = Act2Vec(units=128, input_dim=x_train.shape)

opt = tf.keras.optimizers.Adam(lr=2e-4, decay=2e-11)

act2vec.compile(loss='mse',

             optimizer=opt,

             metrics=['mse'])
# train the act2vec moedl

act2vec.fit(x_train,x_train,

         batch_size=32*6,

         # epochs=4) # for testing

         epochs=n_epochs)

act2vec.save_weights('HAR-act2vec_model.h5')
X_train = act2vec.encoder(x_train)

X_test = act2vec.encoder(x_test)
print('Encoded X_train shape is: ', X_train.shape)

print('Encoded X_test shape is: ', X_test.shape)
plot_latent_space(X_train, y_train, 'UCI_HAR_Xtrain_latentspace')
random_forest_en = rf(X_train, y_train, n_estimators=100) 
print_result(random_forest_en, X_train, y_train, X_test, y_test)
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

plot_confusion_matrix(random_forest_en, X_test, y_test, class_names=LABELS, file_name='HAR_confusionmatrix_en', normalize=True)
random_forest_raw = rf(x_train.reshape(X_train.shape[0], np.prod(x_train.shape[1:])), y_train, n_estimators=100) 
print_result(random_forest_raw, x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:])), y_train, x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:])), y_test)
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

plot_confusion_matrix(random_forest_raw, x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:])), y_test, class_names=LABELS, file_name='HAR_confusionmatrix_raw', normalize=True)