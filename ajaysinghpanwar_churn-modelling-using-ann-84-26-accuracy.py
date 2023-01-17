import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
churn_data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
# Checking the head of the dataframe

churn_data.head()
# Checking info about data
churn_data.info()
churn_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1,  inplace = True)
churn_data.head()
# Checking how many cities are there in Geography column
churn_data['Geography'].unique()
X = churn_data.iloc[:,:-1]
y = churn_data.iloc[:, -1]
# For Geography column
X[['Germany','Spain']] = pd.get_dummies(X['Geography'], drop_first = True)
X.drop('Geography', axis = 1, inplace = True)
# For Gender column
X['Male'] = pd.get_dummies(X['Gender'], drop_first = True)
X.drop('Gender', axis = 1, inplace = True)
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 51)
print(X_train.shape)
print(X_test.shape)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Check scaled data
# print(X_train)
import keras
from keras.models import Sequential 
from keras.layers import Dense
model = Sequential()
# Adding the input layer and first hidden layer
model.add(keras.Input(11,))
model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the third hidden layer
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 10, epochs = 50)
# Predicting the test set results
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)
# Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

acc_score = accuracy_score(y_test, y_pred)
print("Accuracy : ",acc_score)
# Predicting on new data
new_prediction = model.predict(scaler.fit_transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

print(new_prediction)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# Builing the classifier function
def build_classifier():
    model = Sequential()
    model.add(keras.Input(11,))
    model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
# Passing values to KerasClassifier 
model = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 50)
# We are using 10 fold cross validation here
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
# Checking the accuracies
print(accuracies)
# Checking the mean and standard deviation of the accuracies obtained
mean = accuracies.mean()
variance = accuracies.std()
print("Mean : ",mean)
print("Variance : ",variance)
from sklearn.model_selection import GridSearchCV
# Building the classifier function
def build_classifier(optimizer):
    model = Sequential()
    model.add(keras.Input(11,))
    model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
# Passing values to KerasClassifier
model = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 50)
# Using Grid Search CV to getting the best parameters
parameters = {'batch_size': [25, 32],
             'epochs': [100, 150],
             'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = model, param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 5)

grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
print("Best Parameters : ",best_parameters)
print("Best Accuracy : ",best_accuracy)