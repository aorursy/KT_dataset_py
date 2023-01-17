

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier as rf

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier as tree

from sklearn.linear_model import LogisticRegression as log

from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift as shf



from sklearn.model_selection import train_test_split



df = pd.read_csv("../input/mushrooms.csv")

df1 = pd.get_dummies(df)

df1["class"]=df1["class_p"]

df1.drop(["class_e", "class_p"], axis = 1, inplace = True)
x = df1.drop("class", axis = 1)

y = df1["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
clf = rf()

clf.fit(x_train, y_train)
clf.score(x_test, y_test)