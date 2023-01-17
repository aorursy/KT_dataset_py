import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
data = pd.read_csv("/kaggle/input/uci-breast-cancer-wisconsin-original/breast-cancer-wisconsin.data.txt")

data.head()
data.shape
data = data.drop(['1000025'], axis=1)
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    data.index[data[column] == '?'].tolist()

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
print("Numero de registros:"+str(data.shape[0]))

invalid_rows = None

for column in data.columns.values:

    if len(data.index[data[column] == '?'].tolist()) > 0:

        invalid_rows = data.index[data[column] == '?'].tolist()

        data = data.drop(invalid_rows)  

        print(invalid_rows)

        
data.columns = [ "Clump Thickness ", "Uniformity of Cell Size", "Uniformity of Cell Shape ", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]

data.head()
data_vars = data.columns.values.tolist()

Y = ['Class']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="entropy",min_samples_split=10)

tree.fit(X_train,Y_train)
preds = tree.predict(X_test[X])
pd.crosstab(Y_test.values.ravel(),preds,rownames=["Actual"],colnames=["Prediction"])
from sklearn.model_selection  import KFold
cv = KFold(n_splits=20,shuffle=True)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree,data[X],data[Y],scoring="accuracy",cv=cv,n_jobs=1)

score = np.mean(scores)

score
from sklearn.tree import export_graphviz



with open("cancer_dtree.dot","w") as dotfile:

    export_graphviz(tree,out_file=dotfile,feature_names=X)

    dotfile.close()

    

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from graphviz import Source



file = open("cancer_dtree.dot","r")

text = file.read()

Source(text)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000)

forest.fit(data[X],data[Y].values.ravel())
forest.oob_score_