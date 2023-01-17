import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
# Einlesen der Daten
data_train = pd.read_csv("../input/train.csv")
data_test =  pd.read_csv("../input/test.csv")

print(data_train.shape)
print(data_test.shape)

# Zuweisung der Variablen points für die Koordinaten der Punkte, color_category für die Labels der train Daten
color = {0:'blue', 1:'red'}
points = data_train [["X","Y"]].values
color_category = data_train["class"].values

# Visualisierung der train Daten
plt.scatter(points[:,0],points[:,1],s=10,c=data_train["class"].apply(lambda x: color[x]))
# Splitten der train und test Daten, ca. 200 Test 
x_train, x_test, y_train, y_test = train_test_split(points, color_category, test_size = 0.2, random_state = 0)

# Model SVM radial basis function (Support Vector Machine)
# unterteilt Punkte in zwei Klassen mittels Hyperebene
# Bei nichtlinear trennbaren Daten (wie hier): Vektorraum wird in einen höherdimensionalen Raum überführt -> Vektormenge wird linear trennbar
# Bei Rücktransformation in den niedrigerdimensionalen Raum wird lineare Hyperebene zu einer nichtlinearen Hyperfläche
svm_model_rbf = svm.SVC(kernel = "rbf", C = 0.03, gamma='auto')

# Trainieren des Models
svm_model_rbf.fit(x_train,y_train)

print ('Accuracy auf den getesteten train Daten mit SVM: {}'.format(svm_model_rbf.score(x_test, y_test)))
# Zuweisung der Variablen points_test für die Koordinaten der Punkte, color_category_test für die Labels der Testdaten
points_test = data_test [["X","Y"]].values
color_category_test = data_test["class"].values

# Visualisierung der Testdaten
plt.scatter(points_test[:,0],points_test[:,1],s=10,c=data_test["class"].apply(lambda x: color[x]))
# Visualisierung der decision boundary
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Compare
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000'])

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()
    
plot_decision_boundary(svm_model_rbf,x_train,y_train)
# Vorhersage
predictions_svm = svm_model_rbf.predict(points_test)
print (predictions_svm)
print ('Accuracy auf den Testdaten mit SVM: {}'.format(svm_model_rbf.score(points_test, color_category_test)))
# Model k Nearest Neighbour mit 3 Nachbarn
# Klassifikation anhand der 3 nächsten Nachbarpunkte
knn_model = KNeighborsClassifier(3)
knn_model.fit(x_train,y_train)

# Decision Boundary kNN
plot_decision_boundary(knn_model,x_train,y_train)

print ('Accuracy auf den getesteten train Daten mit kNN: {}'.format(knn_model.score(x_test,y_test)))

# Vorhersage der Testdaten
predictions_knn = knn_model.predict(points_test)
print ('Accuracy auf den Testdaten mit kNN: {}'.format(knn_model.score(points_test, color_category_test)))