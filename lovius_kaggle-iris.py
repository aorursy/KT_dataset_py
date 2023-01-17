# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn import datasets as dt , learning_curve ,cross_validation , neighbors , model_selection , preprocessing , metrics , tree , svm , linear_model , naive_bayes , neural_network

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
iris = pd.read_csv("../input/Iris.csv")

iris.head()
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")

plt.show()
target = iris["Species"]

del(iris['Id'])

del(iris['Species'])
x_train,x_test,y_train,y_test=model_selection.train_test_split(iris.values,target.values,random_state=9)

model = neighbors.KNeighborsClassifier(n_neighbors=3);

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('KNeighborsClassifier=',model.score(x_test,y_test))
model = neighbors.RadiusNeighborsClassifier(radius=0.6 , weights='distance')

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('RadiusNeighborsClassifier=',model.score(x_test,y_test))
model = tree.DecisionTreeClassifier(random_state=2)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('DecisionTreeClassifier=',model.score(x_test,y_test))
model = svm.LinearSVC(C=1.0)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('LinearSVC=',model.score(x_test,y_test))
model = linear_model.LogisticRegressionCV(penalty='l1',solver='liblinear')

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('LogisticRegressionCV=',model.score(x_test,y_test))
model = naive_bayes.GaussianNB()

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('GaussianNB=',model.score(x_test,y_test))
model = neural_network.MLPClassifier(max_iter=20000,hidden_layer_sizes=(10,),activation="relu")

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print('MLPClassifier=',model.score(x_test,y_test))