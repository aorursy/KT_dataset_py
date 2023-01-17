import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

from __future__ import absolute_import, division, print_function

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

print('The scikit-learn version is {}.'.format(sklearn.__version__))
#Load the training and test files

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print('training: ', df_train.shape)

print('test: ', df_test.shape)
#Convert to Numpy arrays and separate features/targets

training_samples = df_train.as_matrix()

training_targets = training_samples[:,-1]

training_samples = training_samples[:,1:-1]



test_samples = df_test.as_matrix()

test_samples = test_samples[:,1:]



#Encode the Labels of the categorical data

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# [0:116]

allLabels = np.concatenate( ( training_samples[:, 0:116].flat , test_samples[:, 0:116].flat ) )

le.fit( allLabels )

del allLabels

#print(le.classes_)



#Transform the labels to int values

for colIndex in range(116):

    training_samples[:, colIndex] = le.transform(training_samples[:, colIndex])

    test_samples[:, colIndex] = le.transform( test_samples[:, colIndex] )
#from keras.layers.normalization import BatchNormalization



def larger_model():

    model = Sequential()

    model.add(Dense(1000, input_dim=130, init='normal', activation='relu'))

    #model.add(BatchNormalization())

    #model.add(Activation('relu'))

    #model.add(Dropout(0.5))

    

    model.add(Dense(1000, init='normal', activation='relu'))

    #model.add(BatchNormalization())

    #model.add(Activation('relu'))

    #model.add(Dropout(0.5))

    

    model.add(Dense(500, init='normal', activation='relu'))

    #model.add(BatchNormalization())

    #model.add(Activation('relu'))

    #model.add(Dropout(0.5))

    

    model.add(Dense(100, init='normal', activation='relu'))

    

    model.add(Dense(20, init='normal', activation='relu'))

    #model.add(BatchNormalization())

    #model.add(Activation('relu'))

    #model.add(Dropout(0.5))

    

    model.add(Dense(1, init='normal'))

    #model.add(BatchNormalization())

    #model.add(Activation('relu'))

    #model.add(Dropout(0.5))



    model.compile(loss='mean_absolute_error', optimizer='adam')

    return model
np.random.seed(0)





estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=20, batch_size=50, verbose=1)))

pipeline = Pipeline(estimators)



#Uncomment the following line to run fitting and prediction phases

#The fitting will take a very long time

#pipeline.fit(training_samples, training_targets)

#pred_targets = pipeline.predict(test_samples)
