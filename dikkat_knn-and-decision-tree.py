# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd
data=pd.read_csv("../input/winequality-red.csv")
data.head()
data.shape
#description about the dataset

data.describe()
#infromation about the dataset

data.info()
from matplotlib import pyplot as plt
data['quality'].value_counts().plot.bar()

plt.show()
data['quality'] = data['quality'].map({

        3 : 0,

        4 : 0,

        5 : 0,

        6 : 0,

        7 : 1,

        8 : 1         

})
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
X = data[['fixed acidity','citric acid','residual sugar','sulphates','alcohol']]

Y = data[['quality']]

print(X.shape)

print(Y.shape)
norm=(X-X.min())/(X.max()-X.min())

norm.head()
#KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)  

classifier.fit(norm,Y)
#prediction

knnpred= classifier.predict(norm)

knnpred
#confusion matrix

from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(Y, knnpred))
#accuracy

from sklearn.metrics import accuracy_score 

Accuracy_Score = accuracy_score(Y, knnpred)

Accuracy_Score
#decision tree

# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(norm, Y)
dtpred=classifier.predict(norm)

dtpred
#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y,dtpred)

cm
#accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y,dtpred)

accuracy