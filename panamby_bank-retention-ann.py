# First as always I will import the main libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Now it's time to import the file and check your informations.

dataset = pd.read_csv('../input/Churn_Modelling.csv')

dataset.head()
# Before to start a preprocessing the data I have to import the dataset in the variables.

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# I put in "X" the variable independents and in "y" the variable dependent.
# Notes that I didn't included "RowNumber", "CustomerId" and "Surname", because none of them is relevant for my model.
# As we can see above there are some columns with categorical values.
# Now I will encoding them.
# Let's start to preprocessing our dataset.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
# X_1 refers to "Geography" column.
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
# X_2 refers to "Gender" column.
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
# Let's check if it worked.

X

# I didn't modify the "y" variable, because there is just booleans values and, for my model, is desnecessary encode them.
# Now I will prepare my data before to start my Train and Test set.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# For there isn't any proble with correlations between the values,
# I will aplly the "Feature Scaling".

from  sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Now I will start the more excite part: make the ANN!!!!

# Firt of all: Let's import the Keras libraries and packages.

import keras
from keras.models import Sequential
from keras.layers import Dense
# I will inicializing the ANN.
# As this problem is solved by Classification algorithms, I will call my variable of "classifier".

classifier = Sequential()
# I will add the input layer and the first hidden layer.

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# In "output_dim" I used a not best technique that is get the quantity of output plus input and divide by 2, so 11 + 1 / 2 = 6.
# In "Activiation" I used the best function for hidden layers: "relu".
# Adding the second layer.

classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Note that this time I didn't insert the "input_dim" paramether, because now I Know how many input I have in my ANN.
# Almost finish. Now I will add the output layer.

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# In "output_dim" I put one because I have to guess only the clients who leaves the bank accounts.
# In "activation" I use the best function for it: "sigmoid"
# Compiling the ANN.

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# If I have more than one output, I would have to use in loss paramether other cross entropy.
# Let's do the ANN work for us!!!!

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
# Now our ANN model is done and Let's predict the result.

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Let's make the Confusion Matrix and see if this model is good enought to be delivery to client.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Let's see the result....

cm
# In order to be more easy to see the porcentage, let's do a simple calculus.

(1538 + 187)/2000

# That's good!!!