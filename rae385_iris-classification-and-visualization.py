import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("../input/Iris.csv", index_col = 0)

data.head()
X = data.iloc[:, :4]  #feature matrix

y = data.iloc[:, 4]  #target vector
y.value_counts()
X = X.values   # convert to numpy array for compatibility with other packages

y = y.values
import seaborn as sns  # we dont want to look at all plots individually so we use pair plots

sns.pairplot(data, hue = "Species", height=3) 
#credits to https://www.kaggle.com/skalskip/iris-data-visualization-and-knn-classification

from mpl_toolkits.mplot3d import Axes3D



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

fig = plt.figure(1, figsize=(8, 5))

ax = Axes3D(fig, elev=48, azim=134)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,

           cmap=plt.cm.Set1, edgecolor='k', s = X[:, 3]*50)



for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:

    ax.text3D(X[y == label, 0].mean(),

              X[y == label, 1].mean(),

              X[y == label, 2].mean(), name,

              horizontalalignment='center',

              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),size=15)



ax.set_title("3D visualization", fontsize=20)

ax.set_xlabel("Sepal Length [cm]", fontsize=15)

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("Sepal Width [cm]", fontsize=15)

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("Petal Length [cm]", fontsize=15)

ax.w_zaxis.set_ticklabels([])



plt.show()
from sklearn.model_selection import train_test_split  #used to split to train and test samples

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) #test sample = 0.30 of the data , or 45 samples
from sklearn import tree

iris_classifier = tree.DecisionTreeClassifier()
iris_classifier.fit(x_train, y_train)  # notice the hyperparameters and optimizations below after execution
predictions = iris_classifier.predict(x_test)



from sklearn.metrics import accuracy_score   # another import, this time for accuracy score

print(accuracy_score(y_test,predictions))  # raw score