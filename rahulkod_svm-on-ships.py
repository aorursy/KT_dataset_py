# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# common imports



import numpy as np

import json

import matplotlib.pyplot as plt

file = open('../input/ships-in-satellite-imagery/shipsnet.json')

dataset = json.load(file)

file.close()
dataset.keys()
# write data to numpy arrays

data = np.array(dataset['data']).astype('uint8')
data.shape
# extract label data 



label_data = np.array(dataset['labels']).astype('uint8')
label_data.shape
# reshape data

channels = 3

width = 80

height = 80



X = data.reshape(-1, 3, width, height).transpose([0,2,3,1])

X.shape
# check sample shape and plot

print(X[800].shape)

sample_pic = X[800]

plt.imshow(X[800])
type(sample_pic)
from skimage import color
sample_pic_gr = color.rgb2gray(sample_pic)
sample_pic_gr.shape
plt.imshow(sample_pic_gr)

plt.set_cmap('Greys')
# converting all images to greyscale. Output is a list



X_grey = [ color.rgb2gray(i) for i in X]
X_grey = np.array(X_grey)
X_grey.shape
plt.imshow(X_grey[800])
label_data[800]

# Training data is a 3D matrix. Convert to a 2D matrix. 



X_grey = X_grey.reshape(len(X_grey), -1)
X_grey.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_grey, label_data, test_size=0.2, random_state=42)
# SVM Classifier 



# SVC with rbf kernel. Standard scaler



from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler





svc_clf = Pipeline([

    ('scaler', StandardScaler()),

    ('svc', SVC(gamma='scale'))

])



svc_clf.fit(X_train, y_train)
svc_clf.get_params
from sklearn.metrics import classification_report,accuracy_score

y_pred = svc_clf.predict(X_test)



print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

print('\n')

print(classification_report(y_test, y_pred))
from sklearn.svm import LinearSVC



linearsvc_clf = Pipeline([

    ('scaler', StandardScaler()),

    ('linear_svc', LinearSVC(loss='hinge'))

])



linearsvc_clf.fit(X_train, y_train)
y_pred = linearsvc_clf.predict(X_test)



print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

print('\n')

print(classification_report(y_test, y_pred))
from sklearn.linear_model import SGDClassifier



sgd_clf = Pipeline([

    ('scaler', StandardScaler()),

    ('sgd_clf', SGDClassifier())

])



sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)



print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

print('\n')

print(classification_report(y_test, y_pred))