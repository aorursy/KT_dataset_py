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
dataset= pd.read_csv('../input/churn-predictions-personal/Churn_Predictions.csv')

x= dataset.iloc[:, 3:13].values

y= dataset.iloc[:, 13].values
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()

x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

transformer = ColumnTransformer(transformers=[("onehotencoder", OneHotEncoder(), [1])],

                                remainder='passthrough')

x = transformer.fit_transform(x.tolist())

x = x.astype('float64')

x = x[:, 1:]
#splitting dataset into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.20, random_state= 0)
#feature scalling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
#Creating the Artificial Neural Network



# importing keras and it's packages

import keras

from keras.models import Sequential

from keras.layers import Dense
#Initialising the ANN

classifier = Sequential()
#Adding the input layer and the first hidden layer

classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
#Adding second hidden layer

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
#Adding the output layer

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the ANN into the trainig set

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
#predicting the test set results

y_pred = classifier.predict(x_test)
y_pred
y_pred = (y_pred > 0.5)
y_pred



#Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
