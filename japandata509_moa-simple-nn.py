# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelEncoder

import tensorflow.keras as k

from keras.models import Sequential

import keras
train = pd.read_csv("../input/lish-moa/train_features.csv",index_col='sig_id')

test = pd.read_csv("../input/lish-moa/test_features.csv",index_col='sig_id')

tr_score = pd.read_csv("../input/lish-moa/train_targets_scored.csv",index_col='sig_id')

no_score = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv",index_col='sig_id')

sub = pd.read_csv("../input/lish-moa/sample_submission.csv")
le = LabelEncoder()

for columns in ['cp_type','cp_dose']:

    le.fit(train[columns])

    train[columns] = le.transform(train[columns])

for columns in ['cp_type','cp_dose']:

    le.fit(test[columns])

    test[columns] = le.transform(test[columns])
X = train.to_numpy()

X_test = test.to_numpy()

y = tr_score.to_numpy()

col = tr_score.columns

num_columns=len(X.T)

test_size = len(X_test)
model = Sequential()

model.add(keras.layers.Dense(32,activation='relu',input_shape=(num_columns,)))

model.add(keras.layers.Dense(16,activation='relu'))

model.add(keras.layers.Dense(206,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(X,y,epochs=50,batch_size=512,verbose=1)
test_preds = np.zeros((test.shape[0], y.shape[1]))
for i in range(test_size):

    pred = model.predict(X_test,verbose=1)

    test_preds = pred
test_preds
sub.iloc[:,1:] = test_preds

sub.to_csv("submission.csv",index=False)