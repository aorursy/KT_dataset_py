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
dataset=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
dataset.head()

dataset.describe()
dataset.info()
x=dataset.drop('species',axis=1)
y=dataset['species']
from pandas import read_csv 

from keras.models import Sequential 

from keras.layers import Dense 

from keras.wrappers.scikit_learn import KerasClassifier 

from keras.utils import np_utils 

from sklearn.model_selection import cross_val_score 

from sklearn.model_selection import KFold 

from sklearn.preprocessing import LabelEncoder 

from sklearn.pipeline import Pipeline 
encoder=LabelEncoder()

encoder.fit(y)

encoder_y=encoder.transform(y)
dummy_y = np_utils.to_categorical(encoder_y)
def baseline_model(): 

    # create model

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='relu')) 

    model.add(Dense(3, activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model 
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True) 

results = cross_val_score(estimator, x, dummy_y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
