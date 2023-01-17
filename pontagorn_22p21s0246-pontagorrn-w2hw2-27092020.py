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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
training = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
training.head()
test.head()
training.drop(labels = ["Cabin", "Ticket","Embarked","Name"], axis = 1, inplace = True)
test.drop(labels = ["Cabin", "Ticket","Embarked","Name"], axis = 1, inplace = True)
training.head()
import numpy as np
training[["SibSp","Age","Fare",]] = training[["SibSp","Age","Fare"]].replace(0,np.NaN)
training["SibSp"].fillna(training["SibSp"].median(), inplace = True)
training["Age"].fillna(training["Age"].median(), inplace = True)
training["Fare"].fillna(training["Fare"].median(), inplace = True)
X = training[["SibSp","Parch","Fare","Age"]].to_numpy()
y = training[["Survived"]].to_numpy()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
param_grid = {"max_depth": np.arange(1, 10),
 "criterion":["entropy","gini"]}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(X_train, y_train)
tree.best_estimator_
y_pred = tree.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.naive_bayes import GaussianNB
bmodel = GaussianNB()
result = bmodel.fit(X_train,y_train)
y_pred = bmodel.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
result = mlp_model.fit(X_train,y_train)
y_pred = mlp_model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))