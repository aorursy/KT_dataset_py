# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))





# Prepare data variables

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

SEED = 42 # you can make it anything you want, it's just for consistency of the output 



X = train.drop(labels=['Category', 'Package', 'Malware'], axis=1)

Y = train['Malware']



# Split data variables into train set [X_train(predictors), Y_train(responses)] and validation set (X_val, Y_val)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=SEED)
### KNN slider

### required data: set of predictors(X set), set of responses(Y set)



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score, precision_score

from ipywidgets import interact



@interact

def build_knn(n_neighbours=(1, 30), leaf_size=(10, 40)):

    knn = KNeighborsClassifier(n_neighbors=n_neighbours, leaf_size=leaf_size)

    knn.fit(X=X_train, y=Y_train)



    Y_train_probs = knn.predict_proba(X_train)[:,1]

    Y_pred = knn.predict(X_train)

    

    accuracy = knn.score(X=X_train, y=Y_train)

    precision = precision_score(Y_train, Y_pred)

    auc = roc_auc_score(y_true=Y_train, y_score=Y_train_probs)



    print(f"------- number of neighbours: {n_neighbours}, leaf size: {leaf_size}")

    print(f"accuracy: %.4f" % accuracy)

    print(f"precision: %.4f" % precision)

    print(f"AUC: %.4f" % auc)