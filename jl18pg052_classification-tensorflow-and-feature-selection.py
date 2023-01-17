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
import pandas as pd

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import seaborn as sns

import matplotlib.pyplot as plt
print(tf.__version__)
data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

data.head()
data.shape
data.dtypes
data.isnull().sum()
data["veil-type"].value_counts()
data.drop(columns=["veil-type"],inplace=True)
plt.style.use("ggplot")

data["class"].value_counts().plot(kind="bar", 

                                  figsize = (8,5), color = "darkviolet")

plt.title("Frequency of the classes of our Target variable", size=20)

plt.xlabel("Target Variable", size = 16)

plt.ylabel("Frequency", size = 16)
data.shape
X = data.drop(columns=["class"], axis = 1)

y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
def prepare_inputs(X_train, X_test):

    oe = OrdinalEncoder()

    oe.fit(X_train)

    X_train_enc = oe.transform(X_train)

    X_test_enc = oe.transform(X_test)

    return X_train_enc, X_test_enc
def prepare_targets(y_train, y_test):

    le = LabelEncoder()

    le.fit(y_train)

    y_train_enc = le.transform(y_train)

    y_test_enc = le.transform(y_test)

    return y_train_enc, y_test_enc
def select_features(X_train, y_train, X_test):

    fs = SelectKBest(score_func=chi2, k='all')

    fs.fit(X_train, y_train)

    X_train_fs = fs.transform(X_train)

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)

y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
for i in range(len(fs.scores_)):

    print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores

plt.figure(figsize = (12,6))

plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.title("Feature Importance Score", size = 20)

plt.xlabel("Features/ Variables", size = 16, color = "black")

plt.ylabel("Importance Score", size = 16, color = "black")

plt.show()
X_train.columns
X_new = X.drop(columns = ["cap-surface", "cap-color", "odor", "gill-attachment", "stalk-shape",

                         "stalk-color-above-ring", "stalk-color-below-ring", "veil-color", "ring-number"])
X_dummies = pd.get_dummies(X_new, drop_first=True, columns = ["cap-shape", "bruises",

                                                          "gill-spacing", "gill-size", "gill-color",

                                                          "stalk-root", "stalk-surface-above-ring",

                                                          "stalk-surface-below-ring", "ring-type",

                                                          "spore-print-color", "population", "habitat"])
X_dummies.shape
X_dummies.columns
label = LabelEncoder()

data["class_2"]=label.fit_transform(data["class"])

y_encoded = data["class_2"]
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_dummies, y_encoded, test_size = 0.2, random_state = 0, stratify = y_encoded)
X_train_2.shape
model = Sequential()

model.add(Dense(X_dummies.shape[1], activation = 'relu', input_dim = X_dummies.shape[1]))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train_2, y_train_2, batch_size = 1024, epochs = 15, validation_data=(X_test_2, y_test_2), verbose = 1)
def plotLearningCurve(history,epochs):

  epochRange = range(1,epochs+1)

  plt.figure(figsize = (12,6))

  plt.plot(epochRange,history.history['accuracy'])

  plt.plot(epochRange,history.history['val_accuracy'])

  plt.title('Model Accuracy')

  plt.xlabel('Epoch')

  plt.ylabel('Accuracy')

  plt.legend(['Train','Validation'],loc='upper left')

  plt.show()



  plt.figure(figsize = (12,6))

  plt.plot(epochRange,history.history['loss'])

  plt.plot(epochRange,history.history['val_loss'])

  plt.title('Model Loss')

  plt.xlabel('Epoch')

  plt.ylabel('Loss')

  plt.legend(['Train','Validation'],loc='upper left')

  plt.show()
plotLearningCurve(history,15)
y_pred = model.predict_classes(X_test_2)
cm=confusion_matrix(y_test_2, y_pred)

cm
plt.figure(figsize=(8,6))

sns.set(font_scale=1.2)

sns.heatmap(cm, annot=True, fmt = 'g', cmap="Reds", cbar = False)

plt.xlabel("Predicted Label", size = 18)

plt.ylabel("True Label", size = 18)

plt.title("Confusion Matrix Plotting", size = 20)
print(classification_report(y_test_2, y_pred))