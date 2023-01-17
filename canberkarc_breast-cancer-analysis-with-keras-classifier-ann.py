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
import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno 

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data.head()
msno.matrix(data)
#Eliminate NaN value column

data = data.iloc[:,:32]

data.head()
data.diagnosis.replace(['M','B'],[0,1],inplace = True)

data.head()
id_c = data['id'] 

del data['id']

data.head()
x_data = data.iloc[:,1:]

y_data = data.iloc[:,:1]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.1, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#Import Keras and its packages.

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
classifier = Sequential() #Initialising the ANN

# Let's add input layer and first hidden layer

classifier.add(Dense(units =16, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

classifier.add(Dropout(0.01))

# Let's add 2 more hidden layer and output layer



#2 hidden layers

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.01))



classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.01))



# Output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#Fitting data

classifier.fit(x_train, y_train, epochs = 150, batch_size = 100)
# Predicting the Test set results

y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



print("Classifier accuracy : {}%".format(((cm[0][0] + cm[1][1])/57)*100))
sns.heatmap(cm,annot=True)

plt.savefig('hm.png')