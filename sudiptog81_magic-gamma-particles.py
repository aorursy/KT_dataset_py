import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import pandas_profiling as pp

import matplotlib.pyplot as plt

from colorama import Fore, Style

sns.set()
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/magic-gamma-telescope-dataset/telescope_data.csv')
df.head().T
df.info()
df.profile_report(

    title='Profiling Report for the MAGIC Telescope Dataset'

).to_notebook_iframe()
x = df.drop(['class'], axis=1)

y = df['class']
sns.pairplot(x)

plt.show()
le = LabelEncoder()

y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
scaler = StandardScaler()

x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)
accuracies = {}
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(

    n_estimators=100, 

    criterion='entropy', 

    random_state=0

)

model.fit(x_train, y_train)



y_pred = model.predict(x_test)

sns.heatmap(confusion_matrix(y_test, y_pred), cmap='YlGnBu', annot=True)

plt.show()

accuracies['RandomForest'] = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracies['RandomForest']:.2f}%\n")

print(classification_report(y_test, y_pred, target_names=['gamma', 'hadron']))
from sklearn.linear_model import SGDClassifier



model = SGDClassifier(random_state=0)

model.fit(x_train, y_train)



y_pred = model.predict(x_test)

sns.heatmap(confusion_matrix(y_test, y_pred), cmap='YlGnBu', annot=True)

plt.show()

accuracies['LogReg'] = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracies['LogReg']:.2f}%\n")

print(classification_report(y_test, y_pred, target_names=['gamma', 'hadron']))
from sklearn.svm import SVC



model = SVC(C=57, random_state=0)

model.fit(x_train, y_train)



y_pred = model.predict(x_test)

sns.heatmap(confusion_matrix(y_test, y_pred), cmap='YlGnBu', annot=True)

plt.show()

accuracies['SVM'] = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracies['SVM']:.2f}%\n")

print(classification_report(y_test, y_pred, target_names=['gamma', 'hadron']))
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=0)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

sns.heatmap(confusion_matrix(y_test, y_pred), cmap='YlGnBu', annot=True)

plt.show()

accuracies['DT'] = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracies['DT']:.2f}%\n")

print(classification_report(y_test, y_pred, target_names=['gamma', 'hadron']))
import tensorflow as tf



model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(2, activation='sigmoid'),

])



model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)
x_train_nn, x_cv_nn, y_train_nn, y_cv_nn = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
num_epochs = 6

history = model.fit(

    x_train_nn, y_train_nn, epochs=num_epochs, 

    validation_data=(x_cv_nn, y_cv_nn),

    steps_per_epoch=x_train.shape[0] // num_epochs,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=2)

    ]

)
loss_train = history.history['loss']

loss_validation = history.history['val_loss']

epochs = range(1, num_epochs + 1)

plt.plot(epochs, loss_train, 'g', label='Training')

plt.plot(epochs, loss_validation, 'b', label='Validation')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend()

plt.show()
acc_train = history.history['accuracy']

acc_validation = history.history['val_accuracy']

epochs = range(1, num_epochs + 1)

plt.plot(epochs, acc_train, 'g', label='Training')

plt.plot(epochs, acc_validation, 'b', label='Validation')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy')

plt.legend()

plt.show()
y_pred = model.predict(x_test)

y_pred = [np.argmax(y) for y in y_pred]

sns.heatmap(confusion_matrix(y_test, y_pred), cmap='YlGnBu', annot=True)

plt.show()

accuracies['NN'] = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracies['NN']:.2f}%\n")

print(classification_report(y_test, y_pred, target_names=['gamma', 'hadron']))
ax = sns.barplot(list(accuracies.keys()), list(accuracies.values()))

for p in ax.patches:

    ax.annotate(

        f'{p.get_height():2.2f}%', 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', va = 'center', 

        xytext = (0, -20), textcoords = 'offset points'

    )

plt.xlabel('Models')

plt.ylabel('Accuracy')

plt.show()