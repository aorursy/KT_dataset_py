# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data=pd.read_csv("../input/train.csv")

data.head()

# Any results you write to the current directory are saved as output.
# select only certain columns

# use a "slice" for X, the "feature vectors"

X=data.loc[:,"Pclass":]

# and a single column for y, the "class labels"

y=data["Survived"]

X.head(10)

# Select only the numeric and categorical columns

X=data.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

# convert categorical to "dummy" and then fill NaN

X=pd.get_dummies(X)

X = X.fillna(X.mean())



X.head(10)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Let's try a  decision tree

# It's a classifier, which is an estimateor, which implements fit and predict

from sklearn.tree import DecisionTreeClassifier

DTClf = DecisionTreeClassifier(max_depth=3)

DTClf = DTClf.fit(X_train,y_train)

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test,DTClf.predict(X_test))

metrics.auc(fpr, tpr)

# plot it with "line of no-discrimination"

import matplotlib.pyplot as plt # plotting

plt.figure()

plt.plot(fpr, tpr, color='darkorange')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')



# Export the classifier rules as dot (graphviz) file

# dot -Tpdf tree.dot -o tree.pdf

from sklearn import tree

with open("tree.dot", 'w') as f:

    f = tree.export_graphviz(DTClf, out_file=f, feature_names=X.columns.values.tolist(),

        class_names=["no","yes"], impurity=False, filled=True)


