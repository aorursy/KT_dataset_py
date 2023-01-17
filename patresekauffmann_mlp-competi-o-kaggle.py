# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")




train = train.iloc[:,0:31]

test = test.iloc[:,0:30]



X = train.drop('diagnosis', axis=1)

y = train['diagnosis']




from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

train.diagnosis = labelEncoder.fit_transform(train.diagnosis)

train.head()







# x = atributos

# y = classes

x = train.drop('diagnosis', axis=1)

y = train['diagnosis']



from sklearn.model_selection import train_test_split



# divide o dataset de trainamento em treinamento e teste, separando 25% em teste

# x = atributos e y = classes



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)



x_test.head()
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop







# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))

# Adding dropout to prevent overfitting

classifier.add(Dropout(p=0.1))

# Adding the second hidden layer

classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))

# Adding dropout to prevent overfitting

classifier.add(Dropout(p=0.1))

# Adding the output layer

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=100, nb_epoch=150)