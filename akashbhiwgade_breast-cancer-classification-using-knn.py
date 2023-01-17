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
#import required libraries

import pandas as pd

import numpy as np

from sklearn import neighbors, preprocessing, model_selection



# load the dataset

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')



# drop unnecessary columns which add no value

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)



# extract features and scale them

X = np.array(df.drop('diagnosis', axis=1))

X = preprocessing.scale(X)



# extract labels and map the categorical features for binary classification

y = np.array(df['diagnosis'].map({'M':1, 'B':0}))



# split the dataframe into training and testing data

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)



# create the KNN classifier object

clf = neighbors.KNeighborsClassifier()



# train the classifier, check accuracy and predict the values

clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

pred = clf.predict(X_test)
acc