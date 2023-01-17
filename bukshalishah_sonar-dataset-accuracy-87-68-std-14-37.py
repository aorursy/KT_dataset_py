import numpy as np

import pandas as pd

from keras import layers

from keras import models

from keras import optimizers

from keras import losses

from keras import metrics

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
# Load Data

df = pd.read_csv('../input/sonar.all-data.csv',header=None)

dataset = df.values
# split into input (X) and output (Y) variables

x = dataset[:,0:60].astype(float)



y = dataset[:,60] 
# encode class values as integers



encoder = LabelEncoder()

encoder.fit(y)

encoded_y = encoder.transform(y)
#Build Model



def custom_model():

  

  model = models.Sequential()

  model.add(layers.Dense(32, activation='relu', input_shape=(60,)))

  model.add(layers.Dense(1, activation='sigmoid'))

  

  custom_optimizer = optimizers.RMSprop(lr=0.0035)

  

  model.compile(optimizer=custom_optimizer, loss=losses.binary_crossentropy ,metrics = ['accuracy'])

  

  return model











estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasClassifier(build_fn=custom_model, epochs=25, batch_size=32, verbose=0)))

pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=30, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, x, encoded_y, cv=kfold)

print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))