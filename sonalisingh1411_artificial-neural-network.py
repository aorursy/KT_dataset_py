



#importing the libaries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#Get the dataset

df=pd.read_csv("../input/Churn_Modelling.csv")
#check the dataset

df.head()
#checking the null values

df.isnull().sum()
X = df.iloc[:, 3:13].values

y = df.iloc[:, 13].values
print(X)
print(y)


# Encoding Independent variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Importing keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense
# Initializing the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))



# Adding the second hidden layer

classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))







# Compile the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Predicting the test set results

y_pred = classifier.predict(X_test)



#show if probability of leaving the bank is true or false........lets compare it....... 

y_pred = (y_pred > 0.5)

print(y_pred)



# Making the confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
Accuracy=(1542+142)/2000

Accuracy
Error_rate=(53+263)/2000

Error_rate




from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, init = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 2, n_jobs = 1)

mean = accuracies.mean()

variance = accuracies.std()


