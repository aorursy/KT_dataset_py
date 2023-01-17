# Import some libraries for the information processing and visualization 
# 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
iris_dataset = pd.read_csv("../input/Iris.csv")
target_names = iris_dataset['Species'].unique()
map_to_int = {name: n for n, name in enumerate(target_names)}
targets = np.asarray(iris_dataset['Species'].replace(map_to_int))
# print summary information
iris_dataset.info()
# print data structure
print(iris_dataset.describe())
# Species list in the specific dataset
target_names
# Show first 10 elements of specific dataset 
iris_dataset.head(10)
sns.pairplot(iris_dataset.drop("Id", axis=1), hue="Species", markers=["o", "s", "D"], diag_kind=False)
plt.show()
iris_dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
plt.show()
X = iris_dataset.drop("Id", axis=1).drop("Species", axis=1).values
y = targets

centers = [[1, 1], [-1, -1], [1, -1]]

# Plot the ground truth
fig = plt.figure(1, figsize=(5, 4))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
for name, label in [('Iris-setosa', 0),
                    ('Iris-versicolour', 1),
                    ('Iris-virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()