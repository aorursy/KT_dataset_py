import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dataset = pd.read_csv("../input/zoo.csv")

dataset.head()



# Any results you write to the current directory are saved as output.
X = dataset.iloc[:, 1:17].values

y = dataset.iloc[:, -1].values



# Make number of legs values between 0 & 5

y[:][y[:]==7]=int(0)

X[:,12][X[:,12]==2]=int(1)

X[:,12][X[:,12]==4]=int(2)

X[:,12][X[:,12]==6]=int(3)

X[:,12][X[:,12]==8]=int(4)



y_som = y



# Categorize leg feature

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])

X12 = onehotencoder.fit_transform(X[:, 12].reshape(-1, 1)).toarray()

y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()



y = np.asarray(y, dtype = int)

X12 = np.asarray(X12, dtype = int)

Xnew = np.append(X, X12, axis=1)

X = np.delete(Xnew, 12, axis=1)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

import keras

from keras.models import Sequential

from keras.layers import Dense



# Using an Artificial Neural Network to categorize the data



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 21))



# Adding the second hidden layer

classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))



# Compiling the ANN

classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)

# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_train)

y_pred_test = classifier.predict(X_test)



y_pred_cat = np.argmax(y_pred, axis=1)

y_pred_test_cat = np.argmax(y_pred_test, axis=1)



y_train_cat = np.argmax(y_train, axis=1)

y_test_cat = np.argmax(y_test, axis=1)



#Making the Confusion Matrix (compares actual values with predictions)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_cat, y_pred_cat)

cm_test = confusion_matrix(y_test_cat, y_pred_test_cat)



print(cm)

print(cm_test)
cm_count=0

cm_wrong=0

for i in range(len(cm)):

    cm_count += cm[i,i]

    for v in range(len(cm)):

        cm_wrong += cm[i,v]

cm_wrong -= cm_count

    

cm_test_count=0

cm_test_wrong=0

for i in range(len(cm_test)):

    cm_test_count += cm_test[i,i]

    for v in range(len(cm_test)):

        cm_test_wrong += cm_test[i,v]

cm_test_wrong -= cm_test_count



accuracy = cm_count/(cm_count + cm_wrong)

accuracy_test = cm_test_count/(cm_test_count + cm_test_wrong)



print(accuracy)

print(accuracy_test)



# 100% accuracy classifyihng animals in the train and test sets
