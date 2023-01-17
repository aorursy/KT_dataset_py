# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

from sklearn import datasets #SciKit Learn, loads prepackaged datasets like iris dataset

from sklearn.pipeline import Pipeline #Pipeline defines the flow (DATA --> SCALER --> ALGORITHM --> TRAINING --> PREDICTION)

from sklearn.preprocessing import StandardScaler #imports scaler for model

from sklearn.svm import LinearSVC #imports SVM algorithm

#Fit method creates model, model.fit will do training, model.predict will make predictions



iris = datasets.load_iris()
iris #NumPy array, where each row is one flower

#Target array shows which flowers are of which class

#Target names array associated with 0-2 in target array
X = iris["data"][:, (2, 3)] #petal length, petal width

y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

#y goes through the "target" array in iris dataset and creates a new array with 0 if false and 1 is true 

#if class is Viriginica, put 1, else put 0

svm_clf =  Pipeline([

    ("scalar", StandardScaler()),

    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)) #Linear Support Vector Classifier

    #C = 1 is 

    #loss = hinge, difference between actual values and false values, try to minimize loss

    #random_state, defines starting point for loss function

    #NOTE: look up loss function: aka cost function

])



#Training

svm_clf.fit(X, y)
#Prediction

svm_clf.predict([[2.4, 2.5]])