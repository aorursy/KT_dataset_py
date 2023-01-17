import numpy as np

import pandas as pd

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#plt.style.available
from pylab import rcParams

rcParams['figure.figsize'] = 14,6
from sklearn.cluster import KMeans,AgglomerativeClustering

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn import tree

from sklearn.metrics import confusion_matrix,accuracy_score
data = pd.read_csv("/kaggle/input/survival-from-malignant-melanoma/melanoma.csv")

data.drop(["Unnamed: 0"],axis=1,inplace=True)

data.head()
sns.countplot(x="year",data=data)

plt.show()
sns.countplot(x="year",hue="status",data=data)

plt.show()
sns.countplot(x="ulcer",hue="status",data=data)

plt.show()
sns.countplot(x="status",hue="sex",data=data)

plt.show()
plt.subplot(2,2,1)

plt.scatter(x="age",y="time",c="status",data=data)

plt.xlabel("age")

plt.ylabel("time")

plt.subplot(2,2,2)

plt.scatter(x="age",y="year",c="status",data=data)

plt.xlabel("age")

plt.ylabel("year")

plt.subplot(2,2,3)

plt.scatter(x="year",y="time",c="status",data=data)

plt.xlabel("year")

plt.ylabel("time")

plt.subplot(2,2,4)

plt.scatter(x="thickness",y="time",c="status",data=data)

plt.xlabel("thickness")

plt.ylabel("time")

plt.tight_layout()

plt.show()
plt.subplot(2,2,1)

sns.violinplot(x="status",y="age",data=data)

plt.subplot(2,2,2)

sns.violinplot(x="status",y="year",data=data)

plt.subplot(2,2,3)

sns.violinplot(x="status",y="time",data=data)

plt.subplot(2,2,4)

sns.violinplot(x="status",y="thickness",data=data)

plt.tight_layout()

plt.show()
X=data.drop(["status"],axis=1)

y=data.status

X_train, X_test, y_tain, y_test = train_test_split(X,y,test_size=0.50,random_state=42)
model1 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,

                     weights='uniform')

model1.fit(X_train,y_tain)
print(model1.score(X_train,y_tain))

print(model1.score(X_test,y_test))

model1_train = model1.score(X_train,y_tain)

model1_test = model1.score(X_test,y_test)
pred1 = model1.predict(X_test)
confusion_matrix(y_test,pred1)
mdl1 = accuracy_score(y_test,pred1)
model2 = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=800,

                   multi_class='auto', n_jobs=None, penalty='l2',

                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,

                   warm_start=False)

model2.fit(X_train,y_tain)
print(model2.score(X_train,y_tain))

print(model2.score(X_test,y_test))
pred2 = model2.predict(X_test)
confusion_matrix(y_test,pred2)
mdl2 = accuracy_score(y_test,pred2)
model3 = MLPClassifier(learning_rate_init=0.0001,hidden_layer_sizes=(150,75,50,25),max_iter=3000)

model3.fit(X_train,y_tain)
print(model3.score(X_train,y_tain))

print(model3.score(X_test,y_test))
pred3 = model3.predict(X_test)
confusion_matrix(y_test,pred3)
mdl3 = accuracy_score(y_test,pred3)
comparison = {"accuracy score":[mdl1,mdl2,mdl3]}

comparison = pd.DataFrame(comparison)

comparison.index = ["Model 1(KNeighborsClassifier)","model 2(LogisticRegression)","Model 3(MLPClassifier)"]

comparison
cluster1 = KMeans(n_clusters=3)

cluster1.fit(X)
y1=cluster1.labels_
cluster2 = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',

                        connectivity=None, distance_threshold=None,

                        linkage='ward', memory=None, n_clusters=3,

                        )

cluster2.fit(X)
y2=cluster2.labels_
plt.subplot(2,2,1)

plt.scatter(x="age",y="time",c="status",data=data)

plt.xlabel("age")

plt.ylabel("time")

plt.subplot(2,2,2)

plt.scatter(x="age",y="year",c="status",data=data)

plt.xlabel("age")

plt.ylabel("year")

plt.subplot(2,2,3)

plt.scatter(x="year",y="time",c="status",data=data)

plt.xlabel("year")

plt.ylabel("time")

plt.subplot(2,2,4)

plt.scatter(x="thickness",y="time",c="status",data=data)

plt.xlabel("thickness")

plt.ylabel("time")

plt.tight_layout()

plt.show()
plt.subplot(2,2,1)

plt.scatter(x="age",y="time",c=y1,data=data)

plt.xlabel("age")

plt.ylabel("time")

plt.subplot(2,2,2)

plt.scatter(x="age",y="year",c=y1,data=data)

plt.xlabel("age")

plt.ylabel("year")

plt.subplot(2,2,3)

plt.scatter(x="year",y="time",c=y1,data=data)

plt.xlabel("year")

plt.ylabel("time")

plt.subplot(2,2,4)

plt.scatter(x="thickness",y="time",c=y1,data=data)

plt.xlabel("thickness")

plt.ylabel("time")

plt.tight_layout()

plt.show()
plt.subplot(2,2,1)

plt.scatter(x="age",y="time",c=y2,data=data)

plt.xlabel("age")

plt.ylabel("time")

plt.subplot(2,2,2)

plt.scatter(x="age",y="year",c=y2,data=data)

plt.xlabel("age")

plt.ylabel("year")

plt.subplot(2,2,3)

plt.scatter(x="year",y="time",c=y2,data=data)

plt.xlabel("year")

plt.ylabel("time")

plt.subplot(2,2,4)

plt.scatter(x="thickness",y="time",c=y2,data=data)

plt.xlabel("thickness")

plt.ylabel("time")

plt.tight_layout()

plt.show()