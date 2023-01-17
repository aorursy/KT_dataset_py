import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

raw_data = pd.read_csv('../input/Indian Liver Patient Dataset (ILPD).csv')
raw_data.dropna(axis=0, inplace=True)
raw_data.loc[:, 'gender'] = raw_data['gender'].astype('category')
data = pd.get_dummies(raw_data,prefix=['gender'])

data.head(2)
import matplotlib.pyplot as plt
import matplotlib

fig, ax = plt.subplots(figsize=(16,16))
cax = ax.matshow(raw_data.corr())
ax.set_xticklabels(raw_data.columns)
ax.set_yticklabels(raw_data.columns)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.set_title('Correlation matrix')
fig.colorbar(cax)
plt.show()
#using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

X,y = (data.drop('is_patient', axis=1), data['is_patient'])
hist = []
for i in range(1,10):
    clf = RandomForestClassifier(n_estimators=200, max_depth=i, random_state=0)
    cross_val = cross_val_score(clf, X, y, cv=5)
    hist.append(np.mean(cross_val))
plt.plot(hist)
plt.title('Cross Validations score for RandomForestClassifier')
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import numpy as np

X_normalized = preprocessing.normalize(X, norm='max')
hist = []
for i in range(1,10):
    clf = clf = KNeighborsClassifier(n_neighbors=i)
    cross_val = cross_val_score(clf, X_normalized, y, cv=5)
    hist.append(np.mean(cross_val))
plt.plot(hist)
plt.title('Cross Validations score for KNeighborsClassifier')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
from sklearn.svm import SVC
grid = [0.00001, 0.0001, 0.001, 0.01, 0.1]
hist = []
for val in grid:
    clf = SVC(gamma=val)
    cross_val = cross_val_score(clf, X, y, cv=5)
    hist.append(np.mean(cross_val))
plt.plot([str(i) for i in grid], hist)
plt.title('Cross Validations score for SVC')
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=11, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X,y,validation_split=0.2, epochs=32)
model.evaluate(X,y)
acc = history.history['acc']
loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

fig, ax_l = plt.subplots(1,2, figsize=(16,5))
ax_l[0].plot(acc, 'b',label='Training Accuracy')
ax_l[0].plot(val_acc, 'r',label='Validation Accuracy')
ax_l[0].set_title('Training Accuracy vs Training Loss')
ax_l[0].legend()
ax_l[0].grid()
ax_l[1].plot(loss, 'b',label='Training Loss')
ax_l[1].plot(val_loss, 'r',label='Validation Loss')
ax_l[1].set_title('Validation Accuracy vs Validation Loss')
ax_l[1].legend()
ax_l[1].grid()