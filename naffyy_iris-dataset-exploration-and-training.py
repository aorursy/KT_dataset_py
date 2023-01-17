import numpy as np
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(df.info())
df.head()
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df.plot(y='sepal length (cm)', kind='box', title='Sepal length analysis')

df.plot(y='sepal length (cm)', kind='hist', bins=30, range=(4,8), normed=True, title='Normalized histogram for sepal length')

df.plot(y='sepal length (cm)', kind='hist', bins=30, range=(4,8), cumulative=True, normed=True, title='Cumulative distribution function (CDF) for sepal length')
## Analysing the sepal width
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df.plot(y='sepal width (cm)', kind='box', title='Sepal width analysis')

df.plot(y='sepal width (cm)', kind='hist', bins=30, range=(2,4.5), normed=True, title='Normalized histogram for sepal width')

df.plot(y='sepal width (cm)', kind='hist', bins=30, range=(2,4.5), cumulative=True, normed=True, title='Cumulative distribution function (CDF) for sepal width')
## Statistical EDA
print(df.describe())
print(df.median())
print(df.mean())
print(df.std())

df.plot(kind= 'box') 

df['target'].describe()
df['target'].unique()
from sklearn.model_selection import train_test_split

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print("Data scaled")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
%matplotlib inline

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               c=cmap(idx), marker=markers[idx], label=cl)
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

ppn = Perceptron(max_iter=32, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print('Misclassfied samples: %d' % (y_test != y_pred).sum())
print ('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
from sklearn.svm import SVC

C = 1.0  # SVM regularization parameter
svm = SVC(kernel='linear', C=C, gamma=0.1)
svm.fit(X_train_std, y_train)

print('The accuracy of the linear svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the linear svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train_std, y_train)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
from sklearn.svm import SVC

svm = SVC(kernel='poly', degree=3, C=1.0)
svm.fit(X_train_std, y_train)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X_train_std, y_train)
print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test_std, y_test)))
