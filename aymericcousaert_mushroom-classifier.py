# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble.forest import RandomForestClassifier

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn import tree
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mushrooms = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')  
mushrooms.head()
mushrooms = mushrooms.applymap(lambda x: ord(x))
mushrooms.head()
X = mushrooms.iloc[:, 1:23].values
y = mushrooms.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)
mlp = MLPClassifier(hidden_layer_sizes=(22), activation='logistic', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
results = precision_recall_fscore_support(y_test, predict_test, average='binary',pos_label=101)
print('Precision :', results[0])
print('Recall :', results[1])
print('Medida F1 :', results[2])
print('Error cuadrático medio al finalizar el entrenamiento :', mean_squared_error(y_test, predict_test))
model = RandomForestClassifier(n_estimators = 1)
model.fit(X_train, y_train)
predict_test = model.predict(X_test)
features = list(mushrooms.columns)
features.remove('class')
features = np.array(features)
indice_tree = 0 # Which tree we want to plot
fig, axes = plt.subplots(figsize = (20,20), dpi=800)
tree.plot_tree(model.estimators_[indice_tree],
               feature_names = features, 
               class_names=['edible', 'poisonous'],
               filled = True,
               rounded = True);
plt.bar(features, model.feature_importances_)
plt.xticks(rotation='vertical')
plt.xlabel('features')
plt.ylabel('importancia')
plt.title('importancia de las features')
plt.show()
results_forest = precision_recall_fscore_support(y_test, predict_test, average='binary',pos_label=101)
print('Precision :', results_forest[0])
print('Recall :', results_forest[1])
print('Medida F1 :', results_forest[2])
print('Error cuadrático medio al finalizar el entrenamiento :', mean_squared_error(y_test, predict_test))