import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/Iris.csv")
data_copy = data
data.head()
data.info()
# Check if any duplicate rows are present in the dataset. 
# Checking with Id as it must be unique
data['Id'].duplicated().any()
plt.figure(figsize=[12,12])

# First subplot showing the diamond carat weight distribution
plt.subplot(221)
plt.hist(data['SepalLengthCm'],bins=12,color='darkturquoise')
plt.xlabel('SepalLengthCm')
plt.ylabel('Frequency')
plt.title('Distribution of SepalLengthCm')

# Second subplot showing the diamond depth distribution
plt.subplot(222)
plt.hist(data['SepalWidthCm'],bins=12,color='salmon')
plt.xlabel('SepalWidthCm')
plt.ylabel('Frequency')
plt.title('Distribution of SepalWidthCm')

# Third subplot showing the diamond price distribution
plt.subplot(223)
plt.hist(data['PetalLengthCm'],bins=18,color='skyblue')
plt.xlabel('PetalLengthCm')
plt.ylabel('Frequency')
plt.title('Distribution of PetalLengthCm')

# Fourth subplot showing the diamond price distribution
plt.subplot(224)
plt.hist(data['PetalWidthCm'],bins=15,color='goldenrod')
plt.xlabel('PetalWidthCm')
plt.ylabel('Frequency')
plt.title('Distribution of PetalWidthCm')
central_tendency_SepalLengthCm = {'mean': round(data['SepalLengthCm'].mean(),4), 'median': data['SepalLengthCm'].median(), 'mode': data['SepalLengthCm'].mode()}
central_tendency_SepalLengthCm
central_tendency_SepalWidthCm = {'mean': round(data['SepalWidthCm'].mean(),4), 'median': data['SepalWidthCm'].median(), 'mode': data['SepalWidthCm'].mode()}
central_tendency_SepalWidthCm
central_tendency_PetalLengthCm = {'mean': round(data['PetalLengthCm'].mean(), 4), 'median': data['PetalLengthCm'].median(), 'mode': data['PetalLengthCm'].mode()}
central_tendency_PetalLengthCm
data.groupby(['Species']).size().reset_index(name='counts')
data['Species'].loc[data['PetalLengthCm'] < 2.5].unique()
data['Species'].loc[data['PetalLengthCm'] < 2.5].count()
data['Species'].loc[data['PetalLengthCm'] > 2.5].unique()
data['Species'].loc[data['PetalWidthCm'] < 0.8].unique()
data['Species'].loc[data['PetalWidthCm'] < 0.8].count()
data['Species'].loc[data['PetalWidthCm'] > 0.8].unique()
corr = data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr, mask=mask, vmax=1, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.figure(figsize=(10,6))
sns.kdeplot(data["SepalLengthCm"][data.Species == 'Iris-setosa'], color="darkturquoise", shade=True)
sns.kdeplot(data["SepalLengthCm"][data.Species == 'Iris-versicolor'], color="salmon", shade=True)
sns.kdeplot(data["SepalLengthCm"][data.Species == 'Iris-virginica'], color="skyblue", shade=True)
#plt.xlim(-20,300)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title('Density Plot of SepalLengthCm v/s Species')
plt.show()
plt.figure(figsize=(10,6))
sns.kdeplot(data["SepalWidthCm"][data.Species == 'Iris-setosa'], color="darkturquoise", shade=True)
sns.kdeplot(data["SepalWidthCm"][data.Species == 'Iris-versicolor'], color="salmon", shade=True)
sns.kdeplot(data["SepalWidthCm"][data.Species == 'Iris-virginica'], color="skyblue", shade=True)
#plt.xlim(-20,300)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title('Density Plot of SepalWidthCm v/s Species')
plt.show()
plt.figure(figsize=(10,6))
sns.kdeplot(data["PetalLengthCm"][data.Species == 'Iris-setosa'], color="darkturquoise", shade=True)
sns.kdeplot(data["PetalLengthCm"][data.Species == 'Iris-versicolor'], color="salmon", shade=True)
sns.kdeplot(data["PetalLengthCm"][data.Species == 'Iris-virginica'], color="skyblue", shade=True)
#plt.xlim(-20,300)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title('Density Plot of PetalLengthCm v/s Species')
plt.show()
plt.figure(figsize=(10,6))
sns.kdeplot(data["PetalWidthCm"][data.Species == 'Iris-setosa'], color="darkturquoise", shade=True)
sns.kdeplot(data["PetalWidthCm"][data.Species == 'Iris-versicolor'], color="salmon", shade=True)
sns.kdeplot(data["PetalWidthCm"][data.Species == 'Iris-virginica'], color="skyblue", shade=True)
#plt.xlim(-20,300)
plt.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title('Density Plot of PetalWidthCm v/s Species')
plt.show()
# data_final = pd.get_dummies(data, columns=['Species'])
# data_final.head()
X = data.drop(['Species', 'Id'],1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = GaussianNB() 
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
clf = LogisticRegression()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
clf = tree.DecisionTreeClassifier() 
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
clf = svm.SVC(kernel = 'rbf', C = 10) 
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
clf = KNeighborsClassifier()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
l=list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in l:
    clf=KNeighborsClassifier(n_neighbors=i) 
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    a=a.append(pd.Series(accuracy_score(pred, y_test)))
plt.plot(l, a)
plt.xticks(x)
clf = KNeighborsClassifier(n_neighbors = 4)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
from sklearn import datasets
iris = datasets.load_iris()
X_mod = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X_mod, y, test_size = 0.2)
clf = KNeighborsClassifier()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)
def plot_decision_regions(X_mod, y, clf, test_idx=None, resolution=0.02):
       # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X_mod[:, 0].min() - 1, X_mod[:, 0].max() + 1
    x2_min, x2_max = X_mod[:, 1].min() - 1, X_mod[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X_mod[y == cl, 0], y=X_mod[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
   # highlight test samples
    if test_idx:
       # plot all samples
       X_test, y_test = X_mod[test_idx, :], y[test_idx]
       plt.scatter(X_test[:, 0], X_test[:, 1],
                   c='', edgecolor='black', alpha=1.0,
                   linewidth=1, marker='o',
                   s=100, label='test set')

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_mod=X_combined_std, y=y_combined, clf=clf,test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()