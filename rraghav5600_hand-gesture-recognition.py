import os

import numpy as np

import cv2

from collections import Counter

import random



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
IMG_WIDTH = 18

IMG_HEIGHT = 18



X = []

y = []
def img_process(img):

    thresh_val = 100

    _, img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    img = cv2.GaussianBlur(img, (5,5), 0)

    img = np.reshape(img, (1, -1))[0]

    return img
for dirname, _, filenames in os.walk('/kaggle/input/leapgestrecog/leapgestrecog/leapGestRecog'):

    for filename in filenames:

        img = cv2.imread(os.path.join(dirname, filename),0)

        img = img_process(img)

        X.append(img)

        y.append(int(filename.split('_')[2]))

        if len(X) % 2000 == 0 : print(len(X), 'images processed')
lst = list(zip(X,y))

random.shuffle(lst)

X = [i[0] for i in lst]

y = [i[1] for i in lst]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model_lr = LogisticRegression(solver='lbfgs', multi_class='ovr', n_jobs=10)

model_lr.fit(X_train, y_train)

pred = model_lr.predict(X_test)

print('Accuracy is', accuracy_score(pred, y_test))
model_svm = SVC(gamma='scale')

model_svm.fit(X_train, y_train)

pred = model_svm.predict(X_test)

print('Accuracy is', accuracy_score(pred, y_test))
model_rf = RandomForestClassifier(random_state=1, n_estimators=1000)

model_rf.fit(X_train, y_train)

pred = model_rf.predict(X_test)

print('Accuracy is', accuracy_score(pred, y_test))
model_KNN = KNeighborsClassifier(n_neighbors=3)

model_KNN.fit(X_train, y_train)

pred = model_KNN.predict(X_test)

print('Accuracy is', accuracy_score(pred, y_test))