# unselect the following codes and run them
# %conda install pandas
# %conda install numpy
# %conda install seaborn
# %conda install -c anaconda scikit-learn
# %conda install -c districtdatalabs yellowbrick
# pip install pandas
# pip install numpy
# pip install seaborn
# pip install scikit-learn
# pip install yellowbrick
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
%config InlineBackend.figure_format = 'retina'
# Loading the dataset
iris = datasets.load_iris()
X,y = iris.data,iris.target
X[:5],y[:5]
len(X),len(y)
# converting dataset to a pandas dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris['target']
df
# print(df.to_string())
# pd.set_option('display.height',1000)
# pd.set_option('display.max_rows',500)
# pd.set_option('display.max_columns',500)
# pd.set_option('display.width',1000)
df.describe()
# df['target'].value_counts()
sns.countplot(df['target']);
sns.boxplot(y=df['sepal length (cm)'],x=df['target']);
df.corr()
sns.heatmap(df.corr().abs());
df.corrwith(df["target"]).plot(kind='barh');
# df.corr().abs()['target']
df.isna().sum()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

print(len(df))

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(len(df) - len(X_train))
print(len(df) - len(X_test))
from sklearn.tree import DecisionTreeClassifier, export_graphviz
clf = DecisionTreeClassifier().fit(X_train,y_train)
clf
import graphviz
dtree_viz = export_graphviz(clf, out_file=None,
                            feature_names = df.drop('target', axis=1).columns,
                             filled=True, rounded=True,class_names=['0','1','2'],
                             special_characters=True,
                             impurity=True,proportion=True,
                            node_ids=True)
# Draw graph
graph = graphviz.Source(dtree_viz)
graph
print('Accuracy of classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
X_test[1],y_test[1]
from sklearn.metrics import confusion_matrix
clf.predict(X_test)
y_test
clf.predict(X_test) - y_test
confusion_matrix(y_test,clf.predict(X_test))
from sklearn.metrics import accuracy_score
accuracy_score(y_test,clf.predict(X_test))
# define one new instance
Xnew_input = [[6.7,3.0,5.2,2.3]]
# make a prediction
ynew_predict = clf.predict(Xnew_input)
print("X=%s, Predicted=%s" % (Xnew_input[0], ynew_predict[0]))
# https://machinelearningmastery.com/make-predictions-scikit-learn/
df.tail()
import warnings
warnings.filterwarnings('ignore')
from yellowbrick.contrib.classifier import DecisionViz
viz = DecisionViz(clf, title="Dtree",
                  features=['sepal length (cm)', 'petal length (cm)'],
                  classes=['0', '1','2'])
viz.fit(X_train[:, [0, 1]], y_train)
viz.draw(X_test[:, [0, 1]], y_test)
plt.title('Decision Boundary')
viz.show();
from yellowbrick.model_selection import LearningCurve
visualizer = LearningCurve(clf, scoring="accuracy")
visualizer.fit(X,y)
visualizer.show();
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
km = KMeans(n_clusters=3,verbose=False).fit(X)
km.predict(X)
print(km)
labels = km.labels_
fig = plt.figure(figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30,azim=130)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],c=labels.astype(np.float), edgecolor="r",cmap='viridis')
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means by 3 clusters\n", fontsize=14);
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30,azim=130)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],c=y, edgecolor="r",cmap='viridis')
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("Ground Truth\n", fontsize=14)
plt.show();