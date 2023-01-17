import numpy as np 

import pandas as pd 

dataset = pd.read_csv('../input/Churn_Modelling.csv')

dataset.head()
dataset.info()
dataset = dataset.drop(['RowNumber','CustomerId','Surname'], axis=1)

dataset.head()
geography = pd.get_dummies(dataset['Geography'],drop_first=True)

# similarly for this colimn as well. If there are n dummy columns, consider n-1

gender = pd.get_dummies(dataset['Gender'],drop_first=True)
dataset = dataset.drop(['Geography','Gender'], axis=1)

dataset = pd.concat([dataset,geography,gender],axis=1)

dataset.head()
X = dataset.drop("Exited",axis=1)

y = dataset['Exited']
# Splitting the dataset into the Training set and Test set



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

# Initialising the ANN

classifier = Sequential()
# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))
# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
#Fitting the ANN to the Training set

history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)
#history

# Part 3 - Making predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

y_pred[:5]
# Making the classification report

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test, y_pred)*100)
cm = confusion_matrix(y_test, y_pred)

print(cm)
score = classifier.evaluate(X_test,y_test)

print(score)

print('loss = ', score[0])

print('acc = ', score[1])
# change the epochs to 5, 10 from 2

# we got 79% acc with 2 & 5 & 20 epochs with SGD

# we got 83% acc with 20 epochs with adam

# Initialising the ANN

classifier = Sequential()

classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer

classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

y_pred[:5]


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)

print(cm)
score = classifier.evaluate(X_test,y_test)

print(score)

print('loss = ', score[0])

print('acc = ', score[1])
print(accuracy_score(y_test, y_pred)*100)