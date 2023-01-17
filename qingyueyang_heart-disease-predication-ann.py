import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import os

print(os.listdir("../input"))

df = pd.read_csv("../input/heart.csv")

df.info()
a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['restecg'], prefix = 'restecg')

d = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c, d]

df = pd.concat(frames, axis = 1)

df = df.drop(columns = ['cp', 'thal','restecg', 'slope'])

df.head()
y = df.target.values

X = df.drop(['target'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))

# classifier.add(Dropout(p = 0.1))



# Adding the second hidden layer

classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))

# classifier.add(Dropout(p = 0.1))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# # Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
import seaborn as sns

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred.round())

sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)

#accuracy score

from sklearn.metrics import accuracy_score

ac=accuracy_score(y_test, y_pred.round())

print('accuracy of the model: ',ac)
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))

    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()

variance = accuracies.std()

print('Accuracies are', accuracies,'\n', 'Mean is', mean, '\n', 'Variance is', variance)
# Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

def build_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 23))

    classifier.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],

              'epochs': [100, 200],

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