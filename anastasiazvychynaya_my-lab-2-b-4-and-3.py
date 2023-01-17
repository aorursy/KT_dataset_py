# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from sklearn.neural_network import MLPClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_train_DIR = "../input/chest-xray-pneumonia/chest_xray/train/"
CATEGORIES = ["NORMAL", 'PNEUMONIA']
for category in CATEGORIES:
    path = os.path.join(DATA_train_DIR, category) # path to NORMAL or PNEUMONIA dir
    for img in os.listdir(path): # We are iterating over all images
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # Convert images to array using cv2
        plt.imshow(img_array, cmap = 'gray')
        print('path: ', path, '/', img)
        plt.show()
        break
    break
print('Shape: ', img_array.shape)
# Normalization (in this case we don't need it, but let it be)
IMG_SIZE = 100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
print('path: ', path, '/', img)
plt.show()
# Create training data:
training_data = []
def create_training_data():
    i = 0
    for category in CATEGORIES:
        path = os.path.join(DATA_train_DIR, category) # path to NORMAL or PNEUMONIA dir
        class_num = CATEGORIES.index(category) # 0 it is NORMAL, 1 it is PNEUMONIA
        for img in os.listdir(path): # We are iterating over all images
            if (i < 300 or (i > 1341 and i <= 1641)):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # Convert images to array using cv2
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                    i += 1
                except Exception as exc: # if some images are broken
                    pass
            else:
               i += 1
create_training_data()
print(len(training_data))
# We need to shuffle the data, because all 0 go first and then all 1
import random
random.shuffle(training_data)
# Check it out:
i = 0
for sample in training_data:
    if (i < 10):
        print(sample[1])
        i += 1
    else: break
# As our images are stored as an array, we need to reduce the dimension to use MLPClassifier function:
i = 0
for features in training_data:
    features[0] = features[0].flatten()
    if (i < 5):
        print(features[0])
        i += 1
# Multilayer Perceptron Training (logistic):
X_train = []
Y_train = []
for features, label in training_data:
    X_train.append(features)
    Y_train.append(label)
MLP_Classification_logistic = MLPClassifier(hidden_layer_sizes = 3000, activation = 'logistic', max_iter = 800, random_state = 0).fit(X_train, Y_train)
# Create testing data:
testing_data = []
DATA_test_DIR = "../input/chest-xray-pneumonia/chest_xray/test/"
def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_test_DIR, category) # path to NORMAL or PNEUMONIA dir
        class_num = CATEGORIES.index(category) # 0 it is NORMAL, 1 it is PNEUMONIA
        for img in os.listdir(path): # We are iterating over all images
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # Convert images to array using cv2
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
            except Exception as exc: # if some images are broken
                pass
create_testing_data()
print(len(testing_data))
# As our images are stored as an array, we need to reduce the dimension to use MLPClassifier function:
i = 0
for features in testing_data:
    features[0] = features[0].flatten()
    if (i < 5):
        print(features[0])
        i += 1
X_test = []
Y_test = []
for features, label in testing_data:
    X_test.append(features)
    Y_test.append(label)
# [probability of 0, probability of 1]:
Prob_logistic = MLP_Classification_logistic.predict_proba(X_test)
Prob_logistic
Y_pred = MLP_Classification_logistic.predict(X_test)
print('Accuracy: ', MLP_Classification_logistic.score(X_test, Y_test))
MLP_Classification_logistic.n_layers_
# Multilayer Perceptron Training (relu):
MLP_Classification_relu = MLPClassifier(hidden_layer_sizes = 3000, activation = 'relu', max_iter = 800, random_state = 0).fit(X_train, Y_train)
# [probability of 0, probability of 1]:
Prob_relu = MLP_Classification_relu.predict_proba(X_test)
Prob_relu
Y_pred = MLP_Classification_relu.predict(X_test)
print('Accuracy: ', MLP_Classification_relu.score(X_test, Y_test))
MLP_Classification_relu.n_layers_
# Multilayer Perceptron Training (identity):
MLP_Classification_identity = MLPClassifier(hidden_layer_sizes = 3000, activation = 'identity', max_iter = 800, random_state = 0).fit(X_train, Y_train)
# [probability of 0, probability of 1]:
Prob_identity = MLP_Classification_identity.predict_proba(X_test)
Prob_identity
Y_pred = MLP_Classification_identity.predict(X_test)
print('Accuracy: ', MLP_Classification_identity.score(X_test, Y_test))
MLP_Classification_identity.n_layers_
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # For Mx=0 and Dx=1
from sklearn.svm import SVC
model = make_pipeline(StandardScaler(), SVC(C = 1000000, random_state = 0)) # create SVM classifier
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print('Accuracy (train): ', model.score(X_train, Y_train))
print('Accuracy (test) : ', model.score(X_test, Y_test))
# Let's compare with the case when C = 10 (make the range smaller)
model2 = make_pipeline(StandardScaler(), SVC(C = 10, random_state = 0)) # create SVM classifier
model2.fit(X_train, Y_train)
print('Accuracy (train): ', model2.score(X_train, Y_train))
print('Accuracy (test) : ', model2.score(X_test, Y_test))
# Let's compare with the case when C = 1 (make the range more smaller)
# Nothing changes, when we increase the kernel size from 200 MB to 1000 MB, 
# therefore, I did not change this parameter.
model3 = make_pipeline(StandardScaler(), SVC(C = 1, random_state = 0)) # create SVM classifier
model3.fit(X_train, Y_train)
Y_pred = model3.predict(X_test)
print('Accuracy (train): ', model3.score(X_train, Y_train))
print('Accuracy (test) : ', model3.score(X_test, Y_test))
model4 = make_pipeline(StandardScaler(), SVC(C = 1, kernel = 'rbf', random_state = 0)) # create SVM classifier
model4.fit(X_train, Y_train)
Y_pred = model4.predict(X_test)
print('Accuracy (train): ', model4.score(X_train, Y_train))
print('Accuracy (test) : ', model4.score(X_test, Y_test))
Y_pred
model4 = make_pipeline(StandardScaler(), SVC(C = 1, kernel = 'linear', random_state = 0)) # create SVM classifier
model4.fit(X_train, Y_train)
Y_pred = model4.predict(X_test)
print('Accuracy (train): ', model4.score(X_train, Y_train))
print('Accuracy (test) : ', model4.score(X_test, Y_test))
model4 = make_pipeline(StandardScaler(), SVC(C = 1, kernel = 'poly', degree = 1, random_state = 0)) # create SVM classifier
model4.fit(X_train, Y_train)
Y_pred = model4.predict(X_test)
print('Accuracy (train): ', model4.score(X_train, Y_train))
print('Accuracy (test) : ', model4.score(X_test, Y_test))
# I didn't change the gamma because the default gamma gives better prediction results than 
# when the gamma values are 0.1,1,10,1000000.
model4 = make_pipeline(StandardScaler(), SVC(C = 1, kernel = 'sigmoid', random_state = 0)) # create SVM classifier
model4.fit(X_train, Y_train)
Y_pred = model4.predict(X_test)
print('Accuracy (train): ', model4.score(X_train, Y_train))
print('Accuracy (test) : ', model4.score(X_test, Y_test))
# Lab #3 (Decision Tree. Random Forest)
from sklearn.tree import DecisionTreeClassifier
# random_state = 0, I fix the initial value because otherwise there will be different accuracy every run 
# and it is impossible to analyze anything
DTClassifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, splitter = 'random', min_samples_split = 2, random_state = 0).fit(X_train, Y_train) # create Decision Tree classifier
Y_pred = DTClassifier.predict(X_test)
print('Accuracy (train): ', DTClassifier.score(X_train, Y_train))
print('Accuracy (test) : ', DTClassifier.score(X_test, Y_test))
# We can visualize this
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(DTClassifier, X_test, Y_test, cmap = plt.cm.PuRd)
DTClassifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, splitter = 'random', min_samples_split = 2, random_state = 0).fit(X_train, Y_train) # create Decision Tree classifier
Y_pred = DTClassifier.predict(X_test)
print('Accuracy (train): ', DTClassifier.score(X_train, Y_train))
print('Accuracy (test) : ', DTClassifier.score(X_test, Y_test))
# We can visualize this
plot_confusion_matrix(DTClassifier, X_test, Y_test, cmap = plt.cm.PuRd) # => gini is better!!!
# Why I don't want to make the depth equal to 9?
DTClassifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 9, splitter = 'random', min_samples_split = 2, random_state = 0).fit(X_train, Y_train) # create Decision Tree classifier
Y_pred = DTClassifier.predict(X_test)
print('Accuracy (train): ', DTClassifier.score(X_train, Y_train))
print('Accuracy (test) : ', DTClassifier.score(X_test, Y_test))
# We can visualize this
plot_confusion_matrix(DTClassifier, X_test, Y_test, cmap = plt.cm.PuRd)
from sklearn.ensemble import RandomForestClassifier
RFClassifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 7).fit(X_train, Y_train)
Y_pred = RFClassifier.predict(X_test)
print('Accuracy (train): ', RFClassifier.score(X_train, Y_train))
print('Accuracy (test) : ', RFClassifier.score(X_test, Y_test))
# We can visualize this
plot_confusion_matrix(RFClassifier, X_test, Y_test, cmap = plt.cm.PuRd)