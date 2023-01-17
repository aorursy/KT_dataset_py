# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/StudentsPerformance.csv")
data.info()
data.describe()
data.corr()
data.head()
x = data["reading score"].values.reshape(-1,1)
y = data["writing score"].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3)

linear_reg = LinearRegression()
linear_reg.fit(x_train,y_train)
y_head = linear_reg.predict(x)

print("train score ",linear_reg.score(x_train,y_train))
print("test score ",linear_reg.score(x_test,y_test))

plt.scatter(x,y)
plt.plot(x,y_head,color="orange")
plt.xlabel("reading score")
plt.ylabel("writting score")
plt.show()
new_data = data.drop(["race/ethnicity","parental level of education","test preparation course","lunch"],axis=1)
new_data.gender = [1 if i == "male" else 0 for i in new_data.gender]
x_data = new_data.drop(["gender"],axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
y = new_data.gender.values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3,random_state=1)
score_list = []
score_list2 = []
for i in range(1,21):
    knn_test = KNeighborsClassifier(n_neighbors=i)
    knn_test.fit(x_train,y_train)
    score_list.append(knn_test.score(x_train,y_train))
    score_list2.append(knn_test.score(x_test,y_test))

plt.plot(range(1,21),score_list2)
plt.show()
print(max(score_list2), "   ", score_list2.index(max(score_list2))+1)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print(knn.score(x_test,y_test))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
x = new_data.drop(["gender"],axis=1)
y = new_data.gender.values

pca = PCA(n_components=2, whiten=True)
pca.fit(x,y)
x_pca = pca.transform(x)

print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
new_data["p1"] = x_pca[:,0]
new_data["p2"] = x_pca[:,1]
color = ["blue","red"]

for i in range(2):
    plt.scatter(new_data.p1[new_data.gender == i], new_data.p2[new_data.gender == i], color=color[i])
x = (x-np.min(x))/(np.max(x)-np.min(x))
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.3)
knn = KNeighborsClassifier(n_neighbors=15)
accuracies = cross_val_score(estimator=knn, X=x, y=y, cv=10)

print(np.mean(accuracies))
print(np.std(accuracies))
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train))
print(knn.score(x_test,y_test))
knn = KNeighborsClassifier()
grid = {"n_neighbors":np.arange(1,50)}
knn_cv = GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x_train,y_train)
print(knn_cv.best_params_)
print(knn_cv.best_score_)
y_pred = knn_cv.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))