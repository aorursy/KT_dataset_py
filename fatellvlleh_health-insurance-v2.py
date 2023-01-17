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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
db= pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")
db
from sklearn.tree import DecisionTreeClassifier

db.Gender.replace("Female","1",inplace=True)
db.Gender.replace("Male","1",inplace=True)
db.Vehicle_Age.replace("> 2 Years","1",inplace=True)
db.Vehicle_Age.replace("1-2 Year","0.5",inplace=True)
db.Vehicle_Age.replace("< 1 Year","0",inplace=True)
db.Vehicle_Damage.replace("Yes","1",inplace=True)
db.Vehicle_Damage.replace("No","0",inplace=True)
Y=db.Response
X=db.drop("Response",axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0)
tree = DecisionTreeClassifier(max_depth=2,random_state=0)

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier().fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
