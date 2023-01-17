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
# Input data

df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')



X = df.loc[:, df.columns != 'odor']

y = df['odor'].to_frame()
# Encode features

X_enc = pd.get_dummies(X)



# Standardize features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_std = scaler.fit_transform(X_enc)



# Encode target values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_enc = le.fit_transform(y.values.ravel())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X_std,

    y_enc,

    test_size = 0.2,  # 80/20 training and test split

    #stratify = y_enc,

    random_state = 42

)
# Multiclass as One-Vs-All, using SVM with linear kernel

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt



c_range = np.arange(1.3, 1.4, 0.01) 

c_scores = []

for c in c_range:

    clf = OneVsRestClassifier(estimator=SVC(C=c, kernel='linear'),n_jobs=-1)

    scores = cross_val_score(clf,X_train, y_train, cv=5, scoring='accuracy')

    c_scores.append(scores.mean())

    

plt.plot(c_range, c_scores)

plt.xlabel('Value of C for linear SVM')

plt.ylabel('Cross-Validated Accuracy')

plt.show()

clf_final = OneVsRestClassifier(SVC(C=1.35, kernel='linear'), n_jobs=-1)

clf_final.fit(X_test, y_test)



# output predicted label result

# predict_class = clf_final.predict(X_test)

clf_final.score(X_test, y_test)