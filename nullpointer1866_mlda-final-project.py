import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib # plotting library

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore') # Turn off warnings



def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances.values]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = matplotlib.cm.binary, **options)

    plt.axis("off")
X = pd.read_csv("../input/train.csv")

y = X["label"]

X = X.drop("label", axis = 1)
X.shape
y.shape
%matplotlib inline

rand_digit = X.iloc[37654]

rand_digit_image = rand_digit.values.reshape(28,28)



plt.imshow(rand_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")

plt.axis("off")

plt.show()
y[37654]
y_7 = (y == 7)
y_7[:10]
y[:10]
from sklearn.linear_model import SGDClassifier



classifier = SGDClassifier(random_state = 42, loss = "log")

classifier.fit(X,y_7)
classifier.predict(X.iloc[37654].values.reshape(1,-1))
from sklearn.model_selection import cross_val_predict



y_7_pred = cross_val_predict(classifier, X, y_7, cv = 3)
from sklearn.metrics import confusion_matrix



conf_mx = confusion_matrix(y_7, y_7_pred)

conf_mx
from sklearn.metrics import precision_score, recall_score, f1_score



precision_score(y_7, y_7_pred)*100
recall_score(y_7, y_7_pred)*100
f1_score(y_7, y_7_pred)*100
classifier.fit(X,y)
classifier.predict(X.iloc[37654].values.reshape(1,-1))
classifier.classes_
digit_scores = classifier.decision_function(X.iloc[37654].values.reshape(1,-1))

digit_scores
np.argmax(digit_scores)
y_pred = cross_val_predict(classifier, X, y, cv = 3)

conf_mx = confusion_matrix(y, y_pred)

conf_mx
plt.matshow(conf_mx, cmap = plt.cm.jet)

plt.show()
row_sums = conf_mx.sum(axis = 1, keepdims = True)

norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx, cmap = plt.cm.jet)

plt.show()
class_a, class_b = 8, 5

X_aa = X[(y == class_a) & (y_pred == class_a)]

X_ab = X[(y == class_a) & (y_pred == class_b)]

X_ba = X[(y == class_b) & (y_pred == class_a)]

X_bb = X[(y == class_b) & (y_pred == class_b)]
plt.figure(figsize = (8,8))

plt.subplot(221); plot_digits(X_aa.iloc[:25], images_per_row = 5)

plt.subplot(222); plot_digits(X_ab.iloc[:25], images_per_row = 5)

plt.subplot(223); plot_digits(X_ba.iloc[:25], images_per_row = 5)

plt.subplot(224); plot_digits(X_bb.iloc[:25], images_per_row = 5)
class_a, class_b = 7, 9

X_aa = X[(y == class_a) & (y_pred == class_a)]

X_ab = X[(y == class_a) & (y_pred == class_b)]

X_ba = X[(y == class_b) & (y_pred == class_a)]

X_bb = X[(y == class_b) & (y_pred == class_b)]



plt.figure(figsize = (8,8))

plt.subplot(221); plot_digits(X_aa.iloc[:25], images_per_row = 5)

plt.subplot(222); plot_digits(X_ab.iloc[:25], images_per_row = 5)

plt.subplot(223); plot_digits(X_ba.iloc[:25], images_per_row = 5)

plt.subplot(224); plot_digits(X_bb.iloc[:25], images_per_row = 5)
precision_score(y, y_pred, average = "weighted")*100
recall_score(y, y_pred, average = "weighted")*100
f1_score(y, y_pred, average = "weighted")*100