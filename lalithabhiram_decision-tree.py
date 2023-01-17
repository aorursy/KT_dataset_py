import numpy as np

import pandas as pd
import pandas as pd

Abhi = pd.read_csv("../input/User_Data.csv")
Abhi
X = Abhi.iloc[:,[2]].values

y = Abhi.iloc[:, 4].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
print(clf)
tree.plot_tree(clf.fit(X, y))
y_pred= clf.predict(X_test)

print(y_pred)