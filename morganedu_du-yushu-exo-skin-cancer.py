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
# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline

# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets
from keras.datasets import mnist

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical
df = pd.read_csv('../input/skin-cancer-mnist-ham10000/hmnist_28_28_L.csv')
df.head(10)
X = df.drop(['label'],axis = 1)
y = df.label
X = X/255
n_samples = len(df.index)
image = np.array(X)
image = image.reshape(n_samples,28,28)
plt.figure(figsize=(10,20))
for i in range(0,50) :
    plt.subplot(10,5,i+1)
    plt.axis('off')
    plt.imshow(image[i], cmap="gray_r")
    plt.title(df.label[i])
from sklearn.preprocessing import label_binarize

y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = np.array(X_train).reshape(-1,28,28,1)
X_test = np.array(X_test).reshape(-1,28,28,1)
y_train1 = np.array(y_train)
y_test1 = np.array(y_test)
model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=20, batch_size=200, verbose=1)
scores = model.evaluate(X_test, y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
print(train.history['accuracy'])
print(train.history['val_accuracy'])
def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()
plot_scores(train)
y_cnn = model.predict_classes(X_test)
y = df.label
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
cm = confusion_matrix(y_cnn,y_test)
print(cm)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = np.array(X_train).reshape(-1,28,28,1)
X_test = np.array(X_test).reshape(-1,28,28,1)
model = Sequential()
model.add(Conv2D(28, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(28, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(56, (3, 3), activation='relu'))
model.add(Conv2D(56, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(448, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=50, batch_size=200, verbose=1)
scores = model.evaluate(X_test, y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
plot_scores(train)
model.save('mnist_cnn2.h5')
new_model = load_model('mnist_cnn2.h5')
new_model.summary()
scores = new_model.evaluate(X_test, y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))
for i in range(len(model.layers)):
    print(i, model.layers[i])
for layer in model.layers[10:]:
    layer.trainable = True
for layer in model.layers[0:10]:
    layer.trainable = False
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=50, batch_size=200, verbose=1)
plot_scores(train)
scores = model.evaluate(X_test, y_test1, verbose=0)
print("Score : %.2f%%" % (scores[1]*100))



