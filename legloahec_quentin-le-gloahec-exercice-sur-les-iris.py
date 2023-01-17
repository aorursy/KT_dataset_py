# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
# SeaBorn : librairie de graphiques avancés
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/iris/Iris.csv')
df.columns
df = df.drop(['Id'],axis = 1)
data_train = df.sample(frac=0.8, random_state=1)
data_test = df.drop(data_train.index)
X_train = data_train.drop(['Species'], axis=1)
y_train = data_train['Species']
X_test = data_test.drop(['Species'], axis=1)
y_test = data_test['Species']
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_dtc = dtc.predict(X_test)
print(accuracy_score(y_test, y_dtc))
#0.83, c'est pas fou, nous allons essayer d'obenir un meilleur score
plt.figure(figsize=(30,30))
tree.plot_tree(dtc, feature_names=X_train.columns, class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'], fontsize=14, filled=True)  
dtc1 = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 20)
dtc1.fit(X_train,y_train)
plt.figure(figsize=(30,30))
tree.plot_tree(dtc1, feature_names=X_train.columns, class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'], fontsize=14, filled=True)  
y_dtc1 = dtc1.predict(X_test)
print(accuracy_score(y_test, y_dtc1))
#Avec ce nouvel arbre qu'on va appeler Arbre2, on obtient un score meilleur
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)
print(rf_score)
#comme les deux valeures de l'arbre2 et du meilleur des arbres sont égales on peut en déduire que 
#l'arbre 2 est un des meilleurs arbres.
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12,8))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df.columns[indices])
plt.title('Importance des caracteristiques')
