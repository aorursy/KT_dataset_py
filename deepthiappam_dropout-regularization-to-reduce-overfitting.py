# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file = '../input/sonar.all-data.csv'

data = pd.read_csv(file,header=None)

dataset = data.values

dataset.shape
X = dataset[:,0:60].astype(float)

y = dataset[:,60]
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(y)

y_encoded = encoder.transform(y)
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.optimizers import SGD

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold,cross_val_score

from sklearn.preprocessing import StandardScaler



seed = 7

np.random.seed(seed)



def my_baseline_model():

    model = Sequential()

    model.add(Dense(60,input_dim=60,kernel_initializer='normal',activation='relu'))

    model.add(Dense(30,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))

    sgd = SGD(lr=0.01,momentum=0.8,decay=0,nesterov=False)

    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

    

    return model



estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=my_baseline_model,nb_epoch=100,batch_size=10,verbose=0)))

pipeline = Pipeline(estimators)

kFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)

scores = cross_val_score(pipeline,X,y_encoded,cv=kFold)

print("Accuracy:%.2f%%:(%.2f%%)"%(scores.mean()*100,scores.std()*100))
X[0].shape
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.optimizers import SGD

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold,cross_val_score

from sklearn.preprocessing import StandardScaler



seed = 7

np.random.seed(seed)



def visidropout_model():

    model = Sequential()

    model.add(Dropout(0.2,input_shape=(60,)))

    model.add(Dense(60,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dense(30,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))

    sgd = SGD(lr=0.01,momentum=0.8,decay=0,nesterov=False)

    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

    

    return model



estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=visidropout_model,nb_epoch=100,batch_size=10,verbose=0)))

pipeline = Pipeline(estimators)

kFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)

scores = cross_val_score(pipeline,X,y_encoded,cv=kFold)

print("Accuracy:%.2f%%:(%.2f%%)"%(scores.mean()*100,scores.std()*100))
from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.constraints import maxnorm

from keras.optimizers import SGD

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold,cross_val_score

from sklearn.preprocessing import StandardScaler



seed = 7

np.random.seed(seed)



def visidropout_model():

    model = Sequential()

    model.add(Dropout(0.2,input_shape=(60,)))

    model.add(Dense(60,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dense(30,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))

    sgd = SGD(lr=0.1,momentum=0.9,decay=0,nesterov=False)

    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

    

    return model



estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=visidropout_model,nb_epoch=100,batch_size=10,verbose=0)))

pipeline = Pipeline(estimators)

kFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)

scores = cross_val_score(pipeline,X,y_encoded,cv=kFold)

print("Accuracy:%.2f%%:(%.2f%%)"%(scores.mean()*100,scores.std()*100))
seed = 7

np.random.seed(seed)



def hiddendropout_model():

    model = Sequential()

    model.add(Dropout(0.2,input_shape=(60,)))

    model.add(Dense(60,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(30,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))

    model.add(Dropout(0.2))

    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))

    sgd = SGD(lr=0.1,momentum=0.9,decay=0,nesterov=False)

    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

    

    return model



estimators = []

estimators.append(('standardize',StandardScaler()))

estimators.append(('mlp',KerasClassifier(build_fn=hiddendropout_model,nb_epoch=100,batch_size=10,verbose=0)))

pipeline = Pipeline(estimators)

kFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)

scores = cross_val_score(pipeline,X,y,cv=kFold)

print("Accuracy:%.2f%%:(%.2f%%)"%(scores.mean()*100,scores.std()*100))