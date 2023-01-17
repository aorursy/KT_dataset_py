#Import libraries

import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from keras.layers.normalization import BatchNormalization

from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing.image import ImageDataGenerator
#Load data

train_data = pd.read_csv('../input/digit-recognizer/train.csv')

x_train = train_data.drop(columns = 'label')

y_train = train_data['label']



x_test = pd.read_csv('../input/digit-recognizer/test.csv')
#Look for missing data

print('Train data missing values: %2d' %x_train.isnull().any().sum())

print('Test data missing values: %2d' %x_test.isnull().any().sum())
#Treat data

x_train = x_train.values.astype('float32') / 255

x_test = x_test.values.astype('float32') / 255



x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)



y_train = np_utils.to_categorical(y_train.values)
#Define CNN as function

def cnn_digit(n_pre, n_filters, input_dim, n_dense, n_nodes, drop, n_out, act):

  model = Sequential()

  

  #Preprocessing layers

  for i in range(n_pre):

    if i == 0:

      model.add(Conv2D(filters = n_filters, 

                kernel_size = (3,3), 

                strides = (1,1),

                input_shape = (input_dim[0],input_dim[1],input_dim[2]),

                activation = 'relu'))

    else:

      model.add(Conv2D(filters = n_filters, 

                kernel_size = (3,3), 

                strides = (1,1),

                activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2,2)))

  model.add(Flatten())

  

  #Dense layers

  for i in range(n_dense - 1):

    model.add(Dense(units = n_nodes, activation = 'relu'))

    model.add(Dropout(drop))

  model.add(Dense(units = n_out, activation = act))

  

  #Compilation

  model.compile(loss = 'categorical_crossentropy', 

                optimizer = 'adam',

                metrics = ['accuracy'])

  

  return model
#Test CNN

first_model = cnn_digit(n_pre = 1, 

                        n_filters = 16,

                        input_dim = x_train.shape[1:],

                        n_dense = 1,

                        n_nodes = 32,

                        drop = 0.2,

                        n_out = 10,

                        act = 'softmax')



first_model.fit(x = x_train, y = y_train,

                epochs = 50, batch_size = 128)
#Augment dataset

aug_train = ImageDataGenerator(rotation_range = 7,

                                 horizontal_flip = True,

                                 shear_range = 0.2,

                                 height_shift_range = 0.07,

                                 zoom_range = 0.2)



aug_train = aug_train.flow(x_train, y_train, batch_size = 128)
#Fit augumented model

aug_model = cnn_digit(n_pre = 2,

                      n_filters = 64,

                      input_dim = x_train.shape[1:],

                      n_dense = 2,

                      n_nodes = 128,

                      drop = 0.2,

                      n_out = 10,

                      act = 'softmax')



aug_model.fit_generator(aug_train,  

                        steps_per_epoch = x_train.shape[0]/128,

                        epochs = 150)
#Predict values for test data

pred = [np.argmax(x) for x in aug_model.predict(x_test)]
#Create submission file

sub_file = pd.DataFrame(columns = ['ImageId', 'Label'])

sub_file['ImageId'] = list(range(1,len(x_test)+1))

sub_file['Label'] = pred



sub_file.to_csv('submission.csv', index = False)