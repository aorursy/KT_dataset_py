import pandas as pd

import timeit 

import matplotlib.pyplot as plt

from matplotlib import figure as fig

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC



import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation, Input

from tensorflow.keras.models import Sequential
data_raw = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv', engine='python')

data_raw.head()
#Min-Max Normalizing Data 

data_scaled = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min()) 
data_scaled.head()
ct = pd.crosstab(data_raw['blueFirstBlood'], data_raw['blueWins'],)

ct
ct.plot.bar(stacked=True, figsize =(18.5, 10.5))
# Separating X and y



X = data_scaled.drop(['blueWins', 'gameId'], axis = 1)

y = data_scaled.blueWins.astype(int)

X.head()
# Splitting train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=99)

y_train.head()
# Creating, Training and Testing the Decision Tree Model

dtree = DecisionTreeClassifier(random_state=99)
dtree.fit(X_train, y_train)
predictions_tree = dtree.predict(X_test)

pd.crosstab(y_test, predictions_tree)
accuracy_tree = accuracy_score(y_test, predictions_tree)

print(f'Accuracy of Decision Tree = {accuracy_tree}')
#Creating,Training and Testing the Logistic Regression Model

log_regres = LogisticRegression(random_state=99)
log_regres.fit(X_train, y_train)
predictions_reg = log_regres.predict(X_test)

pd.crosstab(y_test, predictions_reg)
accuracy_reg = accuracy_score(y_test, predictions_reg)

print(f'Accuracy of Logistic Regression Model = {accuracy_reg}')
#Creating,Training and Testing the Support Vector Machine

svm = SVC()
svm.fit(X_train, y_train)
predictions_svm = svm.predict(X_test)

pd.crosstab(y_test, predictions_svm)
accuracy_svm = accuracy_score(y_test, predictions_svm)

print(f'Accuracy of SVM = {accuracy_svm}')
#Defining, Training and Testing a Simple NN



model = Sequential()

model.add(Input(38))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
epochs = 20

batch_size = 16
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
val_loss, val_acc = model.evaluate(X_test, y_test)

print(val_loss)

print(val_acc)
nn_predictions = model.predict_classes(X_test)

nn_predictions
accuracy_nn = accuracy_score(y_test, nn_predictions)

print(f'Accuracy of Neural Network Model = {accuracy_nn}')
#Comparing Prediction Latencies for all methods 

start_dtree = timeit.default_timer()

dtree.predict(X_test)

stop_dtree = timeit.default_timer()



start_svm = timeit.default_timer()

svm.predict(X_test)

stop_svm = timeit.default_timer()



start_lreg = timeit.default_timer()

log_regres.predict(X_test)

stop_lreg = timeit.default_timer()



start_nn = timeit.default_timer()

model.predict_classes(X_test)

stop_nn = timeit.default_timer()



time_dtree = stop_dtree - start_dtree

time_svm = stop_svm - start_svm

time_lreg = stop_lreg - start_lreg

time_nn = stop_nn - start_nn



print(f'Decision Tree Latency : {round(time_dtree * 1000, 4)} milliseconds')

print(f'SVM Latency : {round(time_svm * 1000, 4)} milliseconds')

print(f'Logistic Regression Latency : {round(time_lreg * 1000, 4)} milliseconds')

print(f'Neural Network Latency : {round(time_nn * 1000, 4)} milliseconds')
#Comparing accuracies of all models

print(f'Accuracy of Decision Tree = {round(accuracy_tree * 100, 2)}%')

print(f'Accuracy of SVM = {round(accuracy_svm * 100, 2)}%')

print(f'Accuracy of Logistic Regression Model = {round(accuracy_reg * 100, 2)}%')

print(f'Accuracy of Neural Network Model = {round(accuracy_nn * 100, 2)}%')