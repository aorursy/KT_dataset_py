import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

train = np.genfromtxt('../input/train.csv', delimiter=',')[1:]
target = np.array([i[0] for i in train])
train = np.array([i[1:] for i in train])
plot = {}
for i, j in enumerate(train):
    if not target[i] in plot.keys():
        plot[int(target[i])] = j
fig = plt.figure(figsize=(50,50))
for i in plot.keys():
    fig.add_subplot(1, 10, i + 1)
    plt.imshow(plot[i].reshape(28, 28), cmap='gray')

from sklearn.model_selection import train_test_split
import skimage.transform

train = [skimage.transform.resize(i.reshape(28,28), (10,10)).reshape(1, -1)[0]/255 for i in train]
X_train, X_test, y_train, y_test = train_test_split(train, target)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

params = {
    'weights': ['uniform', 'distance'],
    'n_neighbors': [4]
}

model = GridSearchCV(KNeighborsClassifier(n_jobs=-1), params, verbose=10)
model = model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = model.predict(np.array(X_test))
print('Test - ', accuracy_score(y_test, y_pred))
y_preD = model.predict(X_train)
print('Train - ', accuracy_score(y_train, y_preD))
from sklearn.metrics import confusion_matrix
norm_cm = confusion_matrix(y_test, y_pred)
np.fill_diagonal(norm_cm, 0) # remove correct answers to see errors
plt.imshow(norm_cm, cmap='gray')