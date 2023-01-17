import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

plt.style.use('fivethirtyeight')

%matplotlib inline
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()
iris.info()
iris.describe()
print("The Average Sepal Length (cm) is : ",round(iris['SepalLengthCm'].mean(),2))

print("The Average Sepal Width (cm) is : ",round(iris['SepalWidthCm'].mean(),2))

print("The Average Petal Length (cm) is : ",round(iris['PetalLengthCm'].mean(),2))

print("The Average Petal Width (cm) is : ",round(iris['PetalWidthCm'].mean(),2))
iris.shape
categorical = iris.select_dtypes(include=[np.object])

print("Categorical Columns:",categorical.shape[1])



numerical = iris.select_dtypes(exclude=[np.object])

print("Numerical Columns:",numerical.shape[1])
iris.isnull().any().any()
le = LabelEncoder()

iris['Species'] = le.fit_transform(iris['Species'])
iris['Species']
X = iris.iloc[:, :-1].values

y = iris['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)
scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
error_rate = []



for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred = knn.predict(X_test)

    error_rate.append(np.mean(pred != y_test))
plt.figure(figsize=(12, 6))

plt.plot(range(1,50), error_rate, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid,cv=5)

knn_cv.fit(X,y)
round(knn_cv.best_score_,2)*100
knn_cv.best_params_