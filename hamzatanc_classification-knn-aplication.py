import pandas as pd



iris_data = pd.read_csv("../input/iris-data/iris.csv", index_col  = "id")
iris_data.shape
iris_data.head()
iris_data.corr()
import seaborn as sns

import matplotlib.pyplot as plt



plt.title("SepalLengthCm & SepalWidthCm Scatter Plots by 'Species' - Correlation : - 0.109369")

sns.scatterplot(x = "SepalLengthCm", y = "SepalWidthCm", hue = "Species", data = iris_data);
iris_model_data = iris_data.loc[:,["SepalLengthCm", "SepalWidthCm", "Species"]]

iris_model_data.head()
independent = iris_data.drop("Species", axis = 1)

dependent = iris_data["Species"]
from sklearn.model_selection import train_test_split



independent_train, independent_test, dependent_train, dependent_test = train_test_split(

    independent, 

    dependent, 

    test_size = 0.20, 

    random_state = 50)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn_model = knn.fit(independent_train, dependent_train)



dependent_pred = knn_model.predict(independent_test)
prediction_df = pd.DataFrame({

    "Dependent_Test": dependent_test,

    "Dependent_Predicted": dependent_pred

})



prediction_df
from sklearn.metrics import confusion_matrix



dependent_head_knn = knn.predict(independent_test)

cm_knn = confusion_matrix(dependent_test,dependent_head_knn)



plt.figure(figsize=(20,10))



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap = "Blues",fmt="d",cbar = True, annot_kws={"size": 10})
from sklearn import metrics



print("Accuracy:",metrics.accuracy_score(dependent_test, dependent_pred))
import numpy as np

from sklearn.model_selection import GridSearchCV



knn_params = {"n_neighbors": np.arange(1,20)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 10)

knn_cv.fit(independent_train, dependent_train)



print("Best Score:" + str(knn_cv.best_score_))

print("Best Parameter:" + str(knn_cv.best_params_))

knn = KNeighborsClassifier(n_neighbors = 8)

knn_model = knn.fit(independent_train, dependent_train)



dependent_pred = knn_model.predict(independent_test)
dependent_head_knn = knn.predict(independent_test)

cm_knn = confusion_matrix(dependent_test,dependent_head_knn)



plt.figure(figsize=(20,10))



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap = "Blues",fmt="d",cbar = True, annot_kws={"size": 10})
from matplotlib import pyplot as plt

from sklearn import neighbors, datasets

from matplotlib.colors import ListedColormap



cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



iris = datasets.load_iris()

X = iris.data[:, :2] 

y = iris.target



knn = neighbors.KNeighborsClassifier(n_neighbors = 8)

knn.fit(X, y)



x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1

y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),

                        np.linspace(y_min, y_max, 100))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



plt.figure()

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlabel('sepal length (cm)')

plt.ylabel('sepal width (cm)')

plt.axis('tight')