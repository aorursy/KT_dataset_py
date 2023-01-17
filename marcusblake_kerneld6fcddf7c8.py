# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report


import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainX = pd.read_csv('../input/train.csv')

# #Obtain training labels
trainY = trainX.pop('label')

test = pd.read_csv('../input/test.csv')

trainX.describe()
index = np.random.randint(len(trainX))

print('Label (Number): ', trainY[index])

plt.imshow(np.array(trainX.iloc[index]).reshape((28,28)))

plt.grid(False) 
plt.xticks([])
plt.yticks([])
plt.show()
trainX.isna().values.any()
#Normalize data (i.e normalize rgb color range for pixels from 0-255 to 0-1)
trainX = trainX / 255.0
test = test / 255.0
#Define classifier
classifier = MLPClassifier(solver='adam')
tr_X, val_X, tr_Y, val_Y = train_test_split(trainX, trainY, test_size=0.25) 

#train and evaluate
classifier.fit(tr_X, tr_Y).score(val_X, val_Y)
#train on all data
classifier.fit(trainX, trainY)
#Make prediction on testing data
predictions = classifier.predict(test)
pd.DataFrame(index=pd.Series(np.arange(len(test)) + 1, name='ImageId'),
             data=predictions, columns=['Label']).to_csv('digit_rec.csv')