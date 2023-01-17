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
#Importing important libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style = 'whitegrid'

%matplotlib inline
iris = pd.read_csv("../input/Iris.csv",index_col=0)
iris.head()
sns.pairplot(iris,hue='Species',palette='dark', markers='o')

plt.show()
setosa = iris[iris['Species']=='Iris-setosa']

plt.figure(figsize=(8,5))

sns.kdeplot( setosa['SepalWidthCm'], setosa['SepalLengthCm'],cmap="inferno", shade=True, shade_lowest=False)

plt.show()
sns.violinplot(y='Species', x='SepalLengthCm', data=iris, inner='quartile')

plt.show()

sns.violinplot(y='Species', x='SepalWidthCm', data=iris, inner='quartile')

plt.show()

sns.violinplot(y='Species', x='PetalLengthCm', data=iris, inner='quartile')

plt.show()

sns.violinplot(y='Species', x='PetalWidthCm', data=iris, inner='quartile')

plt.show()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaler.fit(iris.drop('Species',axis=1))
scaled_features = scaler.transform(iris.drop('Species',axis=1))
df = pd.DataFrame(scaled_features,columns=iris.columns[:-1])

df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(scaled_features,iris['Species'],test_size=0.3, random_state=55)
from sklearn.neighbors import KNeighborsClassifier



# Starting with k=1

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



print(confusion_matrix(y_test,pred))

print("\n")

print(classification_report(y_test,pred))
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

accuracy = accuracy_score(y_test,pred)

print("KNN Accuracy: %.2f%%" % (accuracy * 100.0))
error_rate = []



for i in range(1,50):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(15,8))

plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='violet', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

plt.show()
knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=5')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

accuracy = accuracy_score(y_test,pred)

print("KNN Accuracy: %.2f%%" % (accuracy * 100.0))
iris.head()
# Train-test split for SVM, as we don't want the scaled features to put in our SVM model



X_svm = iris.drop('Species',axis=1)

y_svm = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.3,random_state=60)



from sklearn.svm import SVC



svc_model = SVC()

svc_model.fit(X_train,y_train)
predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

accuracy = accuracy_score(y_test,predictions)

print("SVM Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001,0.0001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

print("\n")

print(classification_report(y_test,grid_predictions))
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

accuracy = accuracy_score(y_test,grid_predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))