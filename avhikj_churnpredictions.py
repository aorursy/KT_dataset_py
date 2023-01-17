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
# Installing Tensor Flow

# pip install tensorflow





#Install keras

#pip install -upgrade keras



#Step-1: Data Preprocessing





#Importing the libraries

import matplotlib.pyplot as plt
#Importing Data Set

dataset = pd.read_csv('/kaggle/input/churn-predictions-personal/Churn_Predictions.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values
#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])



onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]
#Splitting the dataset into Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
# Step-2: Creating ANN



#Importing the Keras libraries and packages



import keras

from keras.models import Sequential

from keras.layers import Dense





# Intialising the ANN

classifier = Sequential()



#Adding the input layer and the first hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))



#adding the second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))



#adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))





#Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



#Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 12, epochs = 13)
# Step-3: Making the predictions and evaluationg the model



#Predicting the Test Set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



#Predicting a single new observation

"""Predict if the customer with the following information will leave the bank:

Geography: france

Credit Score: 600

Gender: Male

Age: 40

Tenure: 3

Balance: 60000

Number of Products: 2

HAs Credit Card: Yes

Is Active Member: Yes

Estimated Slary: 50000"""



new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

new_prediction = (new_prediction > 0.5)

new_prediction
#Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
# Improving the ANN

# Dropout Regularization to reduce overfitting if needed



#Turn the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense



def build_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],

             'epochs': [10, 13],

             'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,

                          param_grid = parameters,

                          scoring = 'accuracy',

                          cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
print(best_parameters)

print(best_accuracy)