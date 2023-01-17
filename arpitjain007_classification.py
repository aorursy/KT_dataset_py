# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
labels = (train['label'])

train.drop(labels='label' , inplace=True , axis=1)
X =  train.values.tolist()

#train.drop(train.columns[:-1] , inplace=True , axis=1)
X_test = test.values.tolist()

#test.drop(labels=test.columns[0:-1],inplace=True , axis=1)
some_digit = np.reshape(X[1000], (28,28))

plt.imshow(some_digit)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state=42)

sgd.fit(X , labels)
sgd.predict(X_test)
from sklearn.model_selection import StratifiedKFold

from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3 , shuffle=True , random_state=42)





for train_index  , test_index in skfolds.split(X , labels):

    clone_clf = clone(sgd)

    #print("Train Index :", train_index , "Test Index", test_index)

    

    X_train_folds = train.iloc[train_index]      #using .iloc for selecting indexes

    Y_train_folds = labels.iloc[train_index]      #using .iloc for selecting indexes

    X_test_folds = train.iloc[test_index]        #using .iloc for selecting indexes

    Y_test_folds = labels.iloc[test_index]        #using .iloc for selecting indexes

    

    clone_clf.fit(X_train_folds , Y_train_folds)

    y_pred = (clone_clf.predict(X_test_folds))

    n_correct = sum(y_pred == Y_test_folds)

    print("Accuracy:" , n_correct/len(y_pred))
from sklearn.model_selection import cross_val_score

cross_val_score(sgd , train  , labels , cv=3 , scoring='accuracy')
from sklearn.model_selection import cross_val_predict

pred = cross_val_predict(sgd , train , labels, cv=3)
from sklearn.metrics import confusion_matrix

confusion_matrix(labels , pred)
from sklearn.metrics import precision_score , recall_score

print(precision_score(labels, pred , average="macro"))

print(recall_score(labels, pred , average='macro'))