# importing necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset from drive

data = pd.read_csv("../input/heartbeat/mitbih_train.csv", header=None)
df = pd.DataFrame(data)
df.head()
# showing column wise %ge of NaN values they contains 
null_col = []

for i in df.columns:
  print(i,"\t-\t", df[i].isna().mean()*100)
  if df[i].isna().mean()*100 > 0:
    null_col.append(i)

classes = []
sns.countplot(x=187, data = df) 
class_1 = df[df[187]==1.0]
class_2 = df[df[187]==2.0]
class_3 = df[df[187]==3.0]
class_4 = df[df[187]==4.0]
class_0 = df[df[187]==0.0].sample(n = 8000)
new_df = pd.concat([class_0, class_1, class_2, class_3, class_4])
new_df.head()
sns.countplot(x=187, data = new_df) 
index = 0

fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize=(25,2))

for i in range(5):
  ax[i].plot(new_df[new_df[187]==float(i)].sample(1).iloc[0,:186])
  ax[i].set_title('Class: '+str(i))

#now lets split data in test train pairs

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(new_df.drop([187], axis=1), new_df[187], test_size = 0.1)
X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
from tensorflow.keras import Sequential,utils
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
clf = Sequential()

clf.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (X_train.shape[1],1)))
clf.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu')) 
clf.add(Conv1D(filters=128, kernel_size=(5,), padding='same', activation='relu'))    

clf.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
clf.add(Dropout(0.5))

clf.add(Flatten())

clf.add(Dense(units = 512, activation='relu'))
clf.add(Dense(units = 1024, activation='relu'))

clf.add(Dense(units = 5, activation='softmax'))

clf.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = clf.fit(X_train, y_train, epochs = 10)
# Prediction

y_pred = clf.predict(X_test)
acc = history.history['accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, color='red', label='Training acc')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
from sklearn.metrics import confusion_matrix

y_lbl = [np.where(i == np.max(i))[0][0] for i in y_pred]
mat = confusion_matrix(y_test, y_lbl)
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(mat, annot = True)
# Measure the Accuracy Score

from sklearn import metrics

print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(y_lbl, y_test)))

test_data = pd.read_csv("../input/heartbeat/mitbih_test.csv", header=None)
test_df = pd.DataFrame(test_data)
test_df.head()
classes = []
sns.countplot(x=187, data = test_df) 
index = 0

fig, ax = plt.subplots(nrows = 1, ncols = 5, figsize=(25,2))

for i in range(5):
  ax[i].plot(test_df[test_df[187]==float(i)].sample(1).iloc[0,:186])
  ax[i].set_title('Class: '+str(i))

test_X = test_df.drop([187], axis=1) 
test_y = test_df[187]

test_X = np.array(test_X).reshape(test_X.shape[0], test_X.shape[1], 1)
test_pred_y = clf.predict(test_X)
from sklearn.metrics import confusion_matrix

test_lbl_y = [np.where(i == np.max(i))[0][0] for i in test_pred_y]
mat = confusion_matrix(test_y, test_lbl_y)
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(mat, annot = True)
# Measure the Accuracy Score

from sklearn import metrics

print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(test_lbl_y, test_y)))

