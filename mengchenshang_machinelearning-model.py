import numpy as np

from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



iris =load_iris()

model = KNeighborsClassifier()



np.mean(cross_val_score(model,iris.data, iris.target, cv=5))
from sklearn.svm import SVC



model = SVC()



np.mean(cross_val_score(model,iris.data,iris.target, cv=5))
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()



np.mean(cross_val_score(model,iris.data,iris.target,cv=5))
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt

%matplotlib inline



iris_ = iris.data[:,:2]



model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')

labels = model.fit_predict(iris_)

plt.scatter(iris_[:,0], iris_[:,1], c=labels)