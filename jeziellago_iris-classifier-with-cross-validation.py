import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

import numpy as np # linear algebra

import pandas as pd # data processing
# loading dataset

dataset = pd.read_csv('../input/Iris.csv')

dataset.head()
features = dataset.iloc[:, 1:5].values

labels = dataset.iloc[:,5].values
# encode labels

labelencoder = LabelEncoder()

labels = labelencoder.fit_transform(labels)
def create_model():

    model = Sequential()

    model.add(Dense(units=4, input_dim=4, activation='relu'))

    model.add(Dense(units=3, activation='softmax'))



    model.compile(optimizer=Adam(lr=1e-3), 

                  loss='categorical_crossentropy', 

                  metrics=['categorical_accuracy'])

    return model
kclassifier = KerasClassifier(build_fn=create_model,

                             epochs=1000,

                             batch_size=16)
results = cross_val_score(estimator=kclassifier, 

                          X=features,

                          y=labels,

                          cv=10, 

                          scoring='accuracy')
mean_accuracy = results.mean()

std_accuracy = results.std()



print(f'cross validation accuracy: {mean_accuracy}')

print(f'cross validation std: {std_accuracy}')