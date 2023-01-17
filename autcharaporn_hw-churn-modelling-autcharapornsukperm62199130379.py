import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/bank-churn-modelling/Churn_Modelling.csv')
df.head()
X = df.iloc[:, 6:13].values

y = df.iloc[:, 13].values
print(X)
df.info()
#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
# Fitting K Nearest Neighbor Classification to the Training Set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)  #ดู K=5, มิติ P=2 คือ euclidean space

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) #บ่งบอกว่า testดีหรือไม่ดี true pos,true neg. ควรสูง , False pos.,False neg ควรต่ำ

print(cm)
# Visualising the Training set results



from  matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 

             alpha = 0.75, cmap = ListedColormap(('YELLOW', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)): 

    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1], 

                c = ListedColormap(('red','pink'))(i), label = j)

plt.title('KNN (Training set)')

plt.xlabel('CreditScore')

plt.ylabel('EstimatedSalary')

plt.legend()

plt.show()
# Fitting SVM to the Training Set

from sklearn.svm import SVC

classifier = SVC(kernel = 'poly', degree = 2, random_state = 0) #degree for non-linear

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) #บ่งบอกว่า testดีหรือไม่ดี true pos,true neg. ควรสูง , False pos.,False neg ควรต่ำ

print(cm)
# Visualising the Training set results



from  matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 

             alpha = 0.75, cmap = ListedColormap(('YELLOW', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)): 

    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1], 

                c = ListedColormap(('red','pink'))(i), label = j)

plt.title('KNN (Training set)')

plt.xlabel('CreditScore')

plt.ylabel('EstimatedSalary')

plt.legend()

plt.show()
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) #บ่งบอกว่า testดีหรือไม่ดี true pos,true neg. ควรสูง , False pos.,False neg ควรต่ำ

print(cm)
# Visualising the Training set results



from  matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 

             alpha = 0.75, cmap = ListedColormap(('YELLOW', 'green')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)): 

    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1], 

                c = ListedColormap(('red','pink'))(i), label = j)

plt.title('KNN (Training set)')

plt.xlabel('CreditScore')

plt.ylabel('EstimatedSalary')

plt.legend()

plt.show()