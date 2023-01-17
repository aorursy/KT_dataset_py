# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle
f = open('/kaggle/input/apnea-ecg-pkl/ApneaData.pkl','rb')

data = pickle.load(f)

f.close()
features = []
classes  = []

for row in data:
    features.append(row[:-1])
    classes.append(row[-1])
values, counts = np.unique(classes, return_counts=True)
values, counts
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.pie(counts, labels=['0','1'], colors=['red','green'], autopct='%1.1f%%')
plt.show()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.plot(features[0][1:200])

plt.subplot(2,2,2)
plt.plot(features[1][1:200])

plt.subplot(2,2,3)
plt.plot(features[11000][1:200])

plt.subplot(2,2,4)
plt.plot(features[10000][1:200])

plt.show()
features_np = np.array(features)
features_np = features_np.reshape(features_np.shape[0], features_np.shape[1], 1)
classes_np = np.array(classes)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features_np, classes_np, train_size = 0.8, stratify=classes)
values, counts = np.unique(y_test, return_counts=True)
values, counts
plt.figure(figsize=(5,5))
plt.pie(counts, labels=['0','1'], colors=['red','green'], autopct='%1.1f%%')
plt.show()
values, counts = np.unique(y_train, return_counts=True)
values, counts
plt.figure(figsize=(5,5))
plt.pie(counts, labels=['0','1'], colors=['red','green'], autopct='%1.1f%%')
plt.show()
x_train.shape
import keras
from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Conv1D(64, 40, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=4, strides=2, padding="same"))

model.add(layers.Conv1D(48, 32, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=3, strides=2, padding="same"))

model.add(layers.Conv1D(48, 24, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=2, strides=2, padding="same"))

model.add(layers.Conv1D(32, 16, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool1D(pool_size=2, strides=2, padding="same"))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='sigmoid'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 20, batch_size = 128)
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

y_pred = model.predict_classes(x_test)

# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index   = ['0', '1'], 
                     columns = ['0', '1'])

plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)
plt.title('Apnea Detection Conv1d \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()