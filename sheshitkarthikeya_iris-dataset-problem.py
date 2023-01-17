# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_palette('husl')



from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris_main = pd.read_csv('../input/Iris.csv')
iris_main
iris_main.shape
iris_main.describe
iris_main.info()
iris_main.describe()
iris_main['Species'].value_counts()
tmp = iris_main.drop('Id', axis=1)

g = sns.pairplot(tmp, hue='Species', markers='+')

plt.show()
X = iris_main.drop(['Id','Species'],axis = 1)

Y = iris_main['Species']

X.head()
Y.head()
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.4 , random_state = 5)
K_range  =list(range(1,26))

scores = []

for k in K_range:

    knn = KNeighborsClassifier(n_neighbors = k) 

    knn.fit(X_train,Y_train)

    Y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(Y_test,Y_pred))

plt.plot(K_range , scores)

plt.xlabel('Accuracy scores')

plt.ylabel('value of for knn')

plt.title('accuracy scores with respect to each value of K for knn')

plt.show()
logreg = LogisticRegression()

logreg.fit(X_train , Y_train)

Y_pred = logreg.predict(X_test)

print('the accuracy sore for logistic regressiion is : ',metrics.accuracy_score(Y_test , Y_pred))
knn = KNeighborsClassifier(n_neighbors = 12)

knn.fit(X_train , Y_train)

knn.predict([[2,5,1,1.5]])