import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from cv2 import imread, resize
import os
from sklearn.metrics import classification_report
data_root = '../input/alien-vs-predator-images/data/train'
classes = np.array(os.listdir(data_root))
class_index = dict(zip(classes, [0,1]))
class_index
train_X = []
train_Y = []
for i, j in enumerate(classes):
    folder = data_root + f'/{j}'
    for k, filename in enumerate(os.listdir(folder)):
        img = imread((os.path.join(folder, filename)), 1)
        if k%50==0:
            plt.imshow(img)
            plt.show()
        img = resize(img, (100,100)).flatten()
        train_X.append(img)
        train_Y.append(i)
test_X = []
test_Y = []
for i, j in enumerate(classes):
    folder = '../input/alien-vs-predator-images/data/validation' + f'/{j}'
    print(folder)
    for filename in os.listdir(folder):
        img = imread((os.path.join(folder, filename)), 1)
        img = resize(img, (100,100)).flatten()
        test_X.append(img)
        test_Y.append(i)
clf = MLPClassifier(hidden_layer_sizes=3000, max_iter=300, activation='logistic').fit(train_X, train_Y)
clf.predict_proba(test_X)
pred = clf.predict(test_X)
clf.score(test_X, test_Y)
print(classification_report(test_Y, pred))
clf.n_layers_
clf_1 = MLPClassifier(hidden_layer_sizes=(1000, 200), max_iter=300, activation='relu').fit(train_X, train_Y)
clf_1.score(test_X, test_Y)
print(classification_report(test_Y, clf_1.predict(test_X)))