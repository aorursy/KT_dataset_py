import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting and graphs
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/Iris.csv')
data.head()
X = data.drop(['Id', 'Species'], axis=1).values[:, 2:4]
y = data.Species.values

print("First 5 rows of X:\n", X[:5, :])
print("\nFirst 5 labels in y:\n", y[:5])
markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               c=cmap(idx), marker=markers[idx], label=cl)
sns.countplot(x='Species', data=data)
plt.title("Value Counts of Iris Classes")
plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

print("Now y is numeric!\n", y)
shuffle_index = np.arange(len(y))
np.random.shuffle(shuffle_index)

X_shuffle = X[shuffle_index]
y_shuffle = y[shuffle_index]

y_shuffle
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_shuffle, y_shuffle, test_size=0.3)
print("Training set has {} examples".format(X_train.shape[0]))
print("Test set has {} examples".format(X_test.shape[0]))
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

dt = DecisionTreeClassifier(random_state=0)
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

dt = dt.fit(X_train, y_train)
svm = svm.fit(X_train, y_train)
def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
plot_decision_regions(X_test, y_test, dt)
dt.score(X_test, y_test)
plot_decision_regions(X_test, y_test, svm)
svm.score(X_test, y_test)