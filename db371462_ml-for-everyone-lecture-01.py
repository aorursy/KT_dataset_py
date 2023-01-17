# This command makes sure that we can see plots we create as part of the notebook & without having to save & read files
%matplotlib inline 
import sys
from logging import info

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import datasets, neighbors, preprocessing

# Let's print what versions of the libraries we're using
print(f"python\t\tv {sys.version.split(' ')[0]}\n===")
for lib_ in [np, pd, sns, sklearn, ]:
    sep_ = '\t' if len(lib_.__name__) > 8 else '\t\t'
    print(f"{lib_.__name__}{sep_}v {lib_.__version__}"); del sep_
# The digits dataset
digits = datasets.load_digits()
digits.images.shape
digits.target
def plot_handwritten_digit(the_image, label): # plot_handwritten_digit<-function(the_image, label)
    plt.figure(figsize=(2, 2))
    plt.axis('off')
    plt.imshow(the_image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
# this will show us the pixel values
image_num = 1000
digits.images[image_num]
# and then we can plot them
plot_handwritten_digit(digits.images[image_num], digits.target[image_num])
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, 64))
labels = digits.target
data.shape
data_scaled = preprocessing.scale(data)
data_scaled
data.mean(axis=0)
data_scaled.mean(axis=0)
n_train = int(0.9*n_samples)

X_train = data[:n_train]
y_train = labels[:n_train]
X_test = data[n_train:]
# re-shape this back so we can plot it again as an image
test_images = X_test.reshape((len(X_test), 8, 8))
y_test = labels[n_train:]
X_train.shape
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred_labels = knn.predict(X_test)
pred_labels
pred_probs = knn.predict_proba(X_test)
pred_probs
test_num = 11
plot_handwritten_digit(test_images[test_num], y_test[test_num])
print("true label is %s" % y_test[test_num])
print("predicted label is %s" % pred_labels[test_num])
print("predicted probabilities are %s" % pred_probs[test_num])
np.where(pred_labels != y_test)
test_num = 41
plot_handwritten_digit(test_images[test_num], y_test[test_num])
print("true label is %s" % y_test[test_num])
print("predicted label is %s" % pred_labels[test_num])
print("predicted probabilities are %s" % pred_probs[test_num])
test_num = 43
plot_handwritten_digit(test_images[test_num], y_test[test_num])
print("true label is %s" % y_test[test_num])
print("predicted label is %s" % pred_labels[test_num])
print("predicted probabilities are %s" % pred_probs[test_num])
for test_num in np.where(pred_labels != y_test)[0]:
    print(f"true label is {y_test[test_num]}"
          f"\npredicted label is {pred_labels[test_num]}"
         )
    print("predicted probabilities are %s" % pred_probs[test_num])
    plot_handwritten_digit(test_images[test_num], y_test[test_num])
    plt.show()


