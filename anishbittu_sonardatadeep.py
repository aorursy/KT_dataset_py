# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import read_csv

from keras.models import Sequential 

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler 

from sklearn.pipeline import Pipeline 
dataset=pd.read_csv('../input/sonar-data-set/sonar.all-data.csv')
dataset.head()
dataset.info()
dataset.describe()
x=dataset.drop('R',axis=1)
y=dataset['R']
encoder=LabelEncoder()

encoder.fit(y)

y_encoded=encoder.transform(y)

def create_baseline(): 

    # create model

    model = Sequential() 

    model.add(Dense(60, input_dim=60, activation='relu')) 

    model.add(Dense(30,activation='relu'))

    model.add(Dense(20,activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    return model

def create_smaller():

    # create model 

    model = Sequential() 

    model.add(Dense(30, input_dim=60, activation='relu')) 

    model.add(Dense(1, activation='sigmoid')) 

    # Compile model 

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model 
estimators = [] 

estimators.append(('standardize', StandardScaler())) 

estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators) 

kfold=StratifiedKFold(n_splits=10,shuffle=True)

results=cross_val_score(pipeline,x,y_encoded,cv=kfold)

print("Baseline: %.2f%% (%.2f%%)"%(results.mean()*100,results.std()*100))