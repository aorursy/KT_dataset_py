import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('../input/creditcard.csv')
dataset.head()
from sklearn.preprocessing import StandardScaler
dataset['NormilizedAmount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))
dataset = dataset.drop(['Amount', 'Time'], axis = 1)
X = dataset.iloc[:, dataset.columns != 'Class']
y = dataset.iloc[:, dataset.columns == 'Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)
from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(input_dim = 29, units = 16, activation = 'relu'))
model.add(Dense(units = 24, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(14,  activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 1)
score = model.evaluate(X_test, y_test)
print(score)
y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
cm
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, y_train)
