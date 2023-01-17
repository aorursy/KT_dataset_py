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
# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Importing data

data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

del data['Unnamed: 32']
X = data.iloc[:, 2:].values

y = data.iloc[:, 1].values



# Encoding categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()

y = labelencoder_X_1.fit_transform(y)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)



#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import keras

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Dropout
Classifier = tf.keras.models.Sequential()
# Add the input Layer and first hidden layer



#How to check the number of nodes in hidden layer = [(Number of nodes in input)+(Number of nodes in Output layer)]/2

# (30+1)/2 = 15.5 ~16







# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))

# Adding dropout to prevent overfitting

classifier.add(Dropout(p=0.1))



# Adding the 2nd Hidden Layer

classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))

classifier.add(Dropout(p=0.1))



#Adding the Output Layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=250)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))
sns.heatmap(cm,annot=True)

plt.savefig('h.png')