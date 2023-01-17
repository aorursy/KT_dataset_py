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
df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

df.head()
df.shape
X = df.iloc[:,3:13]

y = df.iloc[:, 13]
X.shape, y.shape
X.Geography.unique()
X['Gender'].unique()
geo_cat = pd.get_dummies(X["Geography"], drop_first = True)

gender_cat = pd.get_dummies(X['Gender'], drop_first = True)
geo_cat.head()
# merge geo_cat and gender_cat into our X.



X = pd.concat([X, geo_cat, gender_cat], axis = 1)
X.head()
X = X.drop(['Geography',"Gender"], axis = 1)
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

X.head()
X_train
import keras

from keras.models import Sequential

from keras.layers import Dense

# from keras.layers import LeakyReLU, PReLU, ELU

# from keras.layers import Dropout
# Initializing the ANN

clf = Sequential()
# Adding the Input layer and the first hidden layer.

clf.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 11 ))

# clf.add(Dense(output_dim = 6, init = 'he_uniform', activation = 'relu', input_dim = 11 )) # Parameter name chaned refer to https://keras.io/api/layers/core_layers/dense/
# Adding the second hidden layer

clf.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu' ))
# Adding Output Layer

clf.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
# Classifier or Model Summary.

clf.summary()
# Compiling the model

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fit the model.

clf_history = clf.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100)
clf_history
clf_history.history.keys()
import matplotlib.pyplot as plt
plt.plot(clf_history.history['accuracy'])

plt.plot(clf_history.history['val_accuracy'])



plt.title('Model Accuracy')



plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.legend(['Train', 'Validation'], loc = 'best')



plt.show()
plt.plot(clf_history.history['loss'])

plt.plot(clf_history.history['val_loss'])



plt.title('Model Loss')



plt.xlabel('Epochs')

plt.ylabel('Loss')



plt.legend(['Train', 'Validation'], loc = 'best')



plt.show()
y_pred = clf.predict(X_test)
y_pred
# Lets set the threshold... if less than 0.5 than set it to false.

y_pred = (y_pred > 0.5)

y_pred
# Lets see the accuracy of our Test Dataset.

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score

score = accuracy_score(y_pred, y_test)



score
clf2 = Sequential()



# Adding the Input layer and the first hidden layer.

clf2.add(Dense(units = 10, kernel_initializer = 'he_normal', activation = 'relu', input_dim = 11 ))





# Adding the second hidden layer

clf2.add(Dense(units = 20, kernel_initializer = 'he_normal', activation = 'relu' ))



# Adding the third hidden layer

clf2.add(Dense(units = 15, kernel_initializer = 'he_normal', activation = 'relu' ))





# Adding Output Layer

clf2.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))



# Compiling the model

clf2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fit the model.

clf2_history = clf2.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100)



y_pred = clf2.predict(X_test)

y_pred = (y_pred > 0.5)

score = accuracy_score(y_pred, y_test)

score
from keras.layers import Dropout
clf3 = Sequential()



# Adding the Input layer and the first hidden layer.

clf3.add(Dense(units = 10, kernel_initializer = 'he_normal', activation = 'relu', input_dim = 11 ))



# Add dropout layer

clf3.add(Dropout(0.3)) # This is just a random threshold as of now



# Adding the second hidden layer

clf3.add(Dense(units = 20, kernel_initializer = 'he_normal', activation = 'relu' ))



# Add dropout layer

clf3.add(Dropout(0.4)) # This is just a random threshold as of now



# Adding the third hidden layer

clf3.add(Dense(units = 15, kernel_initializer = 'he_normal', activation = 'relu' ))



# Add dropout layer

clf3.add(Dropout(0.2)) # This is just a random threshold as of now



# Adding Output Layer

clf3.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))



# Compiling the model

clf3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fit the model.

clf3_history = clf3.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100)
