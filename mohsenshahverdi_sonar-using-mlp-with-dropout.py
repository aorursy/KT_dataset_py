# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import read_csv

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.optimizers import SGD

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# load dataset, split and encode

dataset = read_csv("../input/sonar-data-set/sonar.all-data.csv", header=None)



X = dataset.iloc[:,0:60].values.astype(float)

Y = dataset.iloc[:,60].values



encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)
def create_model():

    # create model

    model = Sequential()

    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu',

    kernel_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(30, kernel_initializer='normal', activation='relu',

    kernel_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model

    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

seed = 7

np.random.seed(seed)

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, batch_size=16,

verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))