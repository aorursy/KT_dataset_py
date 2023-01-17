# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Binary Classification with Sonar Dataset

# Import Libraries

import numpy

from pandas import read_csv

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)
# load dataset and split data 

dataframe = read_csv("../input/sonar-data-set/sonar.all-data.csv", header=None)

dataset = dataframe.values



X = dataset[:,0:60].astype(float)

Y = dataset[:,60]
# encode class values as integers

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)
# create model

def create_model():

	model = Sequential()

	model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))

	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

	# Compile model

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=200, batch_size=15, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
