# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/estadistica/heart.csv')
X = dataset.iloc[:, [0, 6]].values # Edad y frecuencia cardiaca maxima alcanzada
y = dataset.iloc[:, 12].values # Tiene enfermedad cardiaca o no tiene
from sklearn import tree
arbol_clasificacion = tree.DecisionTreeClassifier()
arbol_clasificacion = arbol_clasificacion.fit(X, y)
arbol_clasificacion.score(X,y)
tree.plot_tree(arbol_clasificacion, feature_names=None, class_names=None)
X2 = X[:, 0:6]
print(X2)
arbol_clasificacion2 = tree.DecisionTreeClassifier()
arbol_clasificacion2 = arbol_clasificacion.fit(X2, y)


tree.plot_tree(arbol_clasificacion2, feature_names=None, class_names=None)
plt.show()
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


ax = plot_decision_regions(X2, y, clf=arbol_clasificacion2, legend=2)
plt.xlabel('Edad')
plt.ylabel('Frecuencia cardiaca')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, 
          ['Tiene EC', 'No tiene EC'], 
           framealpha=0.3, scatterpoints=1)

plt.show()
# Dividir entre entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
random_forest_clasificacion = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0)
random_forest_clasificacion.fit(X_train, y_train)
random_forest_clasificacion.score(X_train, y_train)
random_forest_clasificacion.score(X_test, y_test)
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


ax = plot_decision_regions(X, y, clf=random_forest_clasificacion, legend=2)
plt.xlabel('Edad')
plt.ylabel('Frecuencia cardiaca')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, 
          ['no tiene EC', 'tiene EC'], 
           framealpha=0.3, scatterpoints=1)

plt.show()
y_pred = random_forest_clasificacion.predict(X_test)
ax = plot_decision_regions(X_test, y_pred, clf=random_forest_clasificacion, legend=2)
plt.xlabel('Edad')
plt.ylabel('Frecuencia cardiaca')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, 
          ['no tiene EC', 'tiene EC'], 
           framealpha=0.3, scatterpoints=1)

plt.show()
from sklearn.ensemble import RandomForestClassifier
random_forests = RandomForestClassifier(random_state=0)
random_forests.fit(X2, y)
random_forest_clasificacion.score(X2, y)
print(random_forests.feature_importances_)
arbol_1 = random_forests.estimators_[10]
tree.plot_tree(arbol_1, feature_names=None, class_names=None)
plt.show()
arbol_2 = random_forests.estimators_[40]
tree.plot_tree(arbol_2, feature_names=None, class_names=None)
plt.show()
arbol_3 = random_forests.estimators_[75]
tree.plot_tree(arbol_3, feature_names=None, class_names=None)
plt.show()
print(random_forests.predict([[25, 100]]))
print(random_forests.predict([[50, 180]]))
print(random_forests.predict([[73, 80]]))