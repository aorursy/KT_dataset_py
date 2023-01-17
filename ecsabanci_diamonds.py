# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/diamonds/diamonds.csv", index_col = 0)

# index_col = 0 because we dont want to dublicate the indexes

df.head()
df["cut"].unique()
# We need to preserve the order of from best to worst for "cut","clarity" and "color" features....

cut_class_dict = {"Fair":1,"Good":2,"Very Good":3,"Premium":4,"Ideal":5}
clarity_dict = {"I3":1,"I2":2,"I1":3,"SI2":4,"SI1":5,"VS2":6,"VS1":7,"VVS2":8,"VVS1":9,"IF":10,"FL":11}
color_dict = {"J":1,"I":2,"H":3,"G":4,"F":5,"E":6,"D":7}

df["cut"] = df["cut"].map(cut_class_dict)
df["clarity"] = df["clarity"].map(clarity_dict)
df["color"] = df["color"].map(color_dict)

df.head()
import sklearn 
from sklearn import svm, preprocessing

# because of all the dataframe is ordered by price we need to shuffle them all

df = sklearn.utils.shuffle(df)

X = df.drop("price",axis = 1).values
X = preprocessing.scale(X)
y = df["price"].values

# we add the end of X and y ".values" to convert them in an array....


test_size = 200

X_train = X[:-test_size] 
y_train = y[:-test_size]

X_test = X[-test_size:]
y_test = y[-test_size:]

clf = svm.SVR(kernel = "linear")
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
for X,y in zip(X_test,y_test):
    print("Model: {}, Actual: {}".format(clf.predict([X])[0],y))
clf = svm.SVR(kernel = "rbf")
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
for X,y in zip(X_test,y_test):
    print(f"Model: {clf.predict([X])[0]}, Actual: {y}")