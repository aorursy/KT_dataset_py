import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/contraceptive-prevalence-survey/1987 Indonesia Contraception Prevalence Study.csv')

dataset.head()
dataset.info()
dataset.describe().transpose()
sns.pairplot(dataset)
a = dataset.corr()
a
sns.heatmap(a,annot=True)
y = dataset.iloc[:,[-1]]
X = dataset.iloc[:,0:8]
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y)
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
nb1 = MultinomialNB(alpha = 10.0)
nb2 = GaussianNB()
nb3 = BernoulliNB()
nb1.fit(X_train,y_train)
nb2.fit(X_train,y_train)
nb3.fit(X_train,y_train)
print(nb1.score(X_test,y_test))
print(nb2.score(X_test,y_test))
print(nb3.score(X_test,y_test))
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
dtc1 = DecisionTreeClassifier(max_depth=10)
dtc2 = GradientBoostingClassifier(max_depth = 3,learning_rate = 0.1,n_estimators=150)
dtc1.fit(X_train,y_train)
dtc1.score(X_train,y_train)
dtc2.fit(X_train,y_train)
dtc2.score(X_train,y_train)
dtc1.feature_importances_
dtc1.score(X_test,y_test)
dtc2.score(X_test,y_test)
from sklearn.dummy import DummyClassifier
dumb = DummyClassifier(strategy = 'most_frequent')
dumb.fit(X_train,y_train)
xxx = y.values
xx = []
for t in xxx:
    xx.append(t[0])
sns.countplot(xx)
from sklearn.svm import SVC
sv = SVC()
sv.fit(X_train,y_train)
sv.score(X_test,y_test)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [100,100,100])
mlp.fit(X_train,y_train)
mlp.score(X_test,y_test)
