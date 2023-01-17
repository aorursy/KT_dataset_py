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
import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.metrics import *
from keras.models import Sequential,Model
from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
data = pd .read_csv('../input/voice.csv')
data.head(10)
print(data.columns)
label_value_count = data.label.value_counts()
print(label_value_count)
print(data.info())
# Convert string label to float : male = 1, female = 0
dict = {'label':{'male':1,'female':0}}      # label = column name
data.replace(dict,inplace = True)           # replace = str to numerical
x = data.loc[:, data.columns != 'label']
y = data.loc[:,'label']
x.head()
y.head()
x = x.as_matrix()
y = y.as_matrix()
from sklearn.utils import shuffle
x, y = shuffle(x, y, random_state=1010101)
trainX = x[:int(len(x) * 0.7)]
trainY = y[:int(len(y) * 0.7)]
validateX = x[int(len(x) * 0.7) : int(len(x) * 0.9)]
validateY = y[int(len(y) * 0.7) : int(len(y) * 0.9)]
testX = x[int(len(x) * 0.9):]
testY = y[int(len(y) * 0.9):]

print (len(trainX))
print (len(validateX))
print (len(testX))
trainX = np.array(trainX)
trainY = np.array(trainY)
validateX = np.array(validateX)
validateY = np.array(validateY)
testX = np.array(testX)
testY = np.array(testY)
print (trainX.shape)
model = Sequential()
model.add(Dense(len(trainX[0]), input_dim=len(trainX[0]), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=1000, batch_size=1000, validation_data = (validateX, validateY))
predictions = model.predict(testX)
prob = []
limit_prob = []

for k in range(len(predictions)):
    prob.append(round(predictions[k][0], 4))
    if round(predictions[k][0], 4) > 0.5:
        limit_prob.append(1)
    else:
        limit_prob.append(0)
my_submission = pd.DataFrame({'real': testY, 'prediction': prob, 'limit prediction' : limit_prob})
# you could use any filename. We choose submission here
my_submission.to_csv('sample_submission1.csv', index=False)
my_submission.head(50)

