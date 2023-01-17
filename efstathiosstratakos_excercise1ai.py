#Importing some necessary tools

import numpy as np

import pandas as pd

from numpy import genfromtxt
#Reading CSV files into panda

trn = pd.read_csv('../input/titanic/train.csv')

tst = pd.read_csv('../input/titanic/test.csv')

g_s = pd.read_csv('../input/titanic/gender_submission.csv')
#Converting panda to Numpy NDArray

train = trn.to_numpy()

test = tst.to_numpy()

gen_sub = g_s.to_numpy()
#Creating our X and Y training values 

sex_train = train[:,4]

y_train = train[:, 1]

x_train = sex_train[:]=='male'

x_train = x_train.astype(int)

y_train = y_train.astype(int)

x_train = np.array(x_train).reshape(-1,1)

y_train = np.array(y_train).reshape(-1,1)
#Creating our X and Y testing values

sex_test = test[:,3]

x_test = sex_test[:]=='male'

x_test = x_test.astype(int)

y_test = gen_sub[:,1]

y_test = y_test.astype(int)

x_test = np.array(x_test).reshape(-1,1)

y_test = np.array(y_test).reshape(-1,1)
#Binary classification with DT

from sklearn import tree

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB



#Decision Tree Classifier

DT_model = tree.DecisionTreeClassifier()

DT_model = DT_model.fit(x_train, y_train.ravel())

DT_prediction = DT_model.predict(x_test)



#Gaussian Naive Bayes Classifier

GNB_model = GaussianNB()

GNB_model.fit(x_train, y_train.ravel())

GNB_prediction = GNB_model.predict(x_test)



#pred = clf.predict(x_test)

equals = np.array_equal(DT_prediction,GNB_prediction)

print("Decision Tree:", DT_model.score(x_test, y_test))

print("Gaussian Naive Bayes:", GNB_model.score(x_test, y_test))

print("Do they make the same predictions? ", equals)
#Decision Tree Plot

from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(DT_model, out_file=None, filled=True)

graph = graphviz.Source(dot_data)

graph.render('bc')

graph
#Scatter plot for GNB

import matplotlib.pyplot as plt



plt.scatter(x_test, y_test, c=("red"), alpha=0.9)

plt.title("Scatter plot")

plt.xlabel("Sex")

plt.ylabel("Survival")

plt.show()
#Histogram

plt.hist(x_test, bins="auto",histtype="barstacked",color="red", alpha=0.5, label='Sex')

plt.hist(y_test, bins="auto",histtype="step",color="blue", alpha=0.5, label='Survival')

plt.xlabel("Sex")

plt.ylabel("Survival")

plt.legend(loc='upper right')

plt.show()

#    male = 1 || female = 0

#Survived = 1 || passed = 0