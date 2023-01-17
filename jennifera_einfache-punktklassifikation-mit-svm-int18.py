import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
# Methode zum Einlesen der ARFF Datei
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

# Einlesen der train und test Daten
data_train = read_data("../input/kiwhs-comp-1-complete/train.arff")
data_test = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

# print(os.listdir("../input/kiwhs-comp-1-complete/"))

print (data_train)
print (data_test)
# Visualisierung der train Daten
for i in range(len(data_train)):
    point = data_train[i]
    color = "red"
    if point[2] == -1:
        color = "blue"
    plt.scatter(point[0], point[1], s=20, c=color)
# Umformen der Daten von einer Liste in eine tabellenartige Struktur mit Zeilen und Spalten
train = pd.DataFrame(data_train)
print (train.head())

# Zuweisung der Variablen points für die Koordinaten der Punkte, color_category für die Labels
points = train.iloc[:,:2].values
color_category = train.iloc[:,2].values

# Splitten der train und test Daten: 320 train, 80 test
x_train, x_test, y_train, y_test = train_test_split(points, color_category, test_size = 0.2, random_state = 0)

# Zuweisung der Variablen test für die Koordinaten der Testpunkte
test = data_test.iloc[:,1:].values
# Model SVM linear
SVM_model_linear = svm.SVC(kernel = "linear", C = 0.03, gamma='auto')
SVM_model_linear.fit(x_train,y_train)

# Model SVM radial basis function
SVM_model_rbf = svm.SVC(kernel = "rbf", C = 0.03, gamma='auto')
SVM_model_rbf.fit(x_train,y_train)

# Model SVM polynomial
SVM_model_poly = svm.SVC(kernel = "poly", C = 0.03, gamma='auto')
SVM_model_poly.fit(x_train,y_train)

# Model SVM sigmoid
SVM_model_sigmoid = svm.SVC(kernel = "sigmoid", C = 0.03, gamma='auto')
SVM_model_sigmoid.fit(x_train,y_train)

# Vergleich der Accuracy verschiedener Kerneltypen
print ('Accuracy SVM Linear: {}'.format(SVM_model_linear.score(x_test, y_test)))
print ('Accuracy SVM Radial Basis Function: {}'.format(SVM_model_rbf.score(x_test, y_test)))
print ('Accuracy SVM Polynomial: {}'.format(SVM_model_poly.score(x_test, y_test)))
print ('Accuracy SVM Sigmoid: {}'.format(SVM_model_sigmoid.score(x_test, y_test)))
# Visualisierung der decision boundary, Quelle: Discussion der Competition, Farben rot für 1, blau für -1
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
# Visualisierung der decision boundary vom linear SVM
plot_decision_boundary(SVM_model_linear,x_train,y_train)
# Visualisierung der decision boundary vom rbf SVM
plot_decision_boundary(SVM_model_rbf,x_train,y_train)
# Visualisierung der decision boundary vom polynomial SVM
plot_decision_boundary(SVM_model_poly,x_train,y_train)
# Visualisierung der decision boundary vom sigmoid SVM
plot_decision_boundary(SVM_model_sigmoid,x_train,y_train)
# Vorhersage der Testdaten
predictions_svm_linear = SVM_model_linear.predict(test)
predictions_svm_rbf= SVM_model_rbf.predict(test)
predictions_svm_poly = SVM_model_poly.predict(test)
predictions_svm_sigmoid = SVM_model_sigmoid.predict(test)

# Methode zur Visualisierung der Vorhersage
def plotPrediction(prediction):
    for i in range(len(prediction)):
        point = test[i]
        color = "red"
        if prediction[i] == -1:
            color = "blue"
        plt.scatter(point[0], point[1], s=20, c=color)
# Visualisierung der Vorhersage vom linear SVM
plotPrediction(predictions_svm_linear)
# Visualisierung der Vorhersage vom rbf SVM
plotPrediction(predictions_svm_rbf)
# Visualisierung der Vorhersage vom polynomial SVM
plotPrediction(predictions_svm_poly)
# Visualisierung der Vorhersage vom sigmoid SVM
plotPrediction(predictions_svm_sigmoid)
# Submissions erstellen
submission_svm_linear = pd.DataFrame({"Id (String)": list(range(0,len(predictions_svm_linear))), "Category (String)": predictions_svm_linear.astype(int)})
submission_svm_rbf = pd.DataFrame({"Id (String)": list(range(0,len(predictions_svm_rbf))), "Category (String)": predictions_svm_rbf.astype(int)})
submission_svm_poly = pd.DataFrame({"Id (String)": list(range(0,len(predictions_svm_poly))), "Category (String)": predictions_svm_poly.astype(int)})
submission_svm_sigmoid = pd.DataFrame({"Id (String)": list(range(0,len(predictions_svm_sigmoid))), "Category (String)": predictions_svm_sigmoid.astype(int)})

submission_svm_linear.to_csv("submission_svm_linear", index=False, header=True)
submission_svm_rbf.to_csv("submission_svm_rbf", index=False, header=True)
submission_svm_poly.to_csv("submission_svm_poly", index=False, header=True)
submission_svm_sigmoid.to_csv("submission_svm_sigmoid", index=False, header=True)