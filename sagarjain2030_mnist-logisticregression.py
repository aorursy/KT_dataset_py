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
data = pd.read_csv('../input/train.csv')

data = np.array(data)

data
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

scaler_Object = MinMaxScaler()

X = data[:,1:]

y = data[:,:1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

X_train
y_train
scaler_Object.fit(X_train)

scaled_X_train = scaler_Object.transform(X_train)

scaled_X_test = scaler_Object.transform(X_test)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()

logisticRegr.fit(scaled_X_train, y_train)
from sklearn.metrics import confusion_matrix,classification_report

predictions = logisticRegr.predict(scaled_X_test)

accu = logisticRegr.score(scaled_X_test,y_test)

predictions
accu
test_data = pd.read_csv('../input/test.csv')

test_data = np.array(test_data)

test_data
scaled_test_data = scaler_Object.transform(test_data)

scaled_test_data
predictions_test_data = logisticRegr.predict(scaled_test_data)

predictions_test_data
data_to_submit = pd.DataFrame(columns=['Label'])

data_to_submit['Label'] = predictions_test_data

data_to_submit.insert(0, 'ImageID', range(1, 1 + len(data_to_submit)))

data_to_submit

data_to_submit.to_csv('csv_to_submit.csv', index = False)