import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# we set a random seed so something random won't affect our results

np.random.seed(2)
data = pd.read_csv('../input/creditcard.csv')
data.head()
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Amount', 'Time'], axis=1)
data.head()
sns.countplot(data['Class'])
X = data.iloc[:, data.columns != 'Class']

Y = data.iloc[:, data.columns == 'Class']
X.corrwith(data.Class).plot.bar(figsize=(20, 10), fontsize=12, grid=True)
plt.figure(figsize=(20, 10))

sns.heatmap(data.corr(), annot= True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train.shape
model = Sequential()

model.add(Dense(units=16, input_dim= 29, activation='relu'))

model.add(Dense(units=24, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=20, activation='relu'))

model.add(Dense(units=24, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=15, epochs=5, validation_split=0.2)
model.evaluate(X_test, Y_test)
y_pred = model.predict(X_test)
cm_matrix = confusion_matrix(Y_test, y_pred.round())
print(cm_matrix)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train.values.ravel())
y_pred = random_forest.predict(X_test)
random_forest.score(X_test, Y_test)
cm_matrix = confusion_matrix(Y_test, y_pred.round())
print(cm_matrix)
fraud_indices = np.array(data[data.Class == 1].index)

non_frud_indices = data[data.Class == 0].index

num_fraud = len(fraud_indices)

print(num_fraud)
random_normal = np.array(np.random.choice(non_frud_indices, num_fraud, replace=False))

print(len(random_normal))
undersample_idx = np.concatenate([fraud_indices, random_normal])
new_data = data.iloc[undersample_idx, :]
X_under = new_data.iloc[:, new_data.columns != 'Class']

Y_under = new_data.iloc[:, new_data.columns == 'Class']



X_train, X_test, Y_train, Y_test = train_test_split(X_under, Y_under, test_size=0.3, random_state=0)
# due to the small amount of data we use random forests instead of neural network and decrease number of estimators to improve performance



random_forest = RandomForestClassifier(n_estimators=10)



random_forest.fit(X_train, Y_train.values.ravel())

y_pred = random_forest.predict(X_test)

print(random_forest.score(X_test, Y_test))



cm_matrix = confusion_matrix(Y_test, y_pred.round())

print(cm_matrix)
X_over, Y_over = SMOTE().fit_sample(X, Y.values.ravel())
X_train, X_test, Y_train, Y_test = train_test_split(X_over, Y_over, test_size=0.3, random_state=0)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=15, epochs=5, validation_split=0.2)
model.evaluate(X_test, Y_test)

y_pred = model.predict(X_test)



cm_matrix = confusion_matrix(Y_test, y_pred.round())

print(cm_matrix)