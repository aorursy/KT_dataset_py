# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
dataset
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 1].values
print(X)
print(y)
# Label Encoding the "m and b" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)
#making those values btw -1 and 3 for easy calculation
#splitting the model into test and train set and froming cnn layers 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)




# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#i have used threshhold 50%
#like it gives 1 if y_pred is greater than 0.5 
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#so basically 63 says its correct as 0 and 46 says its correct as 1
#knowing the test set value before
X_test[0]
y_test[0]
# simply trying on x_test[0]
k=ann.predict([[-0.20656118,  0.28631105, -0.13712355, -0.27925989,  1.01337588,
        0.80655631,  0.69932048,  0.84606465,  1.11127916,  1.48173507,
       -0.05259361, -0.51936216,  0.11234263, -0.14668714, -0.54234829,
       -0.15806338,  0.08707975,  0.25042949, -0.42284231,  0.07946914,
        0.02915933,  0.64857047,  0.17987034, -0.06360678,  1.09727399,
        0.83547382,  1.14378486,  1.37791231,  1.10695714,  1.49368807]])
#add dummy variables,then u have to scale by sc object
if(k>0.5):
    print("diagnosis is M")
else:
    print("diagnosis is B")