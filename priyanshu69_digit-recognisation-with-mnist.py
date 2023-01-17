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
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
combined = pd.concat((train.loc[:,'pixel0':'pixel783'],

                    test.loc[:,'pixel0':'pixel783']))
def compress(df):

    for col in df.columns:

        df[col] = df[col].astype('uint8')
import time

print('MEMORY USAGE BEFORE COMPRESSING :' + str(combined.memory_usage().sum()/1024) + 'KB')

tic = time.time()

compress(combined)

toc = time.time()

print('MEMORY USAGE AFTER COMPRESSING :' + str(combined.memory_usage().sum()/1024) + 'KB')

print('time taken :' + str(toc-tic) + 'sec')
combined.isnull().sum().sum()

# No missing values, we can start building our model 
from sklearn.model_selection import cross_val_score

def score(model,X_train,y):

    score = cross_val_score(model,X_train,y,scoring = 'accuracy',cv=5)

    return score.mean()
X_train = combined[:len(train)]

y = train['label']

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

print('CV score of RF model :' + str(score(clf,X_train,y)))

clf.fit(X_train,y)

print('training score of RF model :' + str((y == clf.predict(X_train)).mean()))
from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
dummy_y = np_utils.to_categorical(y)
dummy_y.shape
def neu_net():

    model = Sequential()

    # creating model

    model.add(Dense(786,input_dim = 784, activation = 'relu'))

    model.add(Dense(256, activation = 'relu'))

    model.add(Dense(100, activation = 'relu'))

    model.add(Dense(10,activation = 'softmax'))

    # compiling model

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
estimator = KerasClassifier(build_fn=neu_net, epochs = 30, batch_size = 50, verbose = 2)
estimator.fit(X_train,y)
predictions = estimator.predict(test, verbose = 2)
clf.fit(X_train,y)

sample_submission['Label'] = clf.predict(test)
sample_submission['Label'] = predictions