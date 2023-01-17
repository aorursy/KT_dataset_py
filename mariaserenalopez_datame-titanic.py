# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Algebra lineal
import numpy as np 

# Procesamiento de Datos
import pandas as pd 

# Visualizaci'on de datos
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algoritmos de prediccion
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn. ensemble import VotingClassifier

import warnings
warnings.filterwarnings('ignore')
test = pd.read_csv('../input/titanicdatame/TestCleanData2_Titanic.csv')
train = pd.read_csv('../input/titanicdatame/TrainCleanData2_Titanic.csv')
# Todos los datos son numericos
train.info()
# Analisis descriptivo
train.describe()
# Columnas del set de datos train
# 11 variables independientes y una variable dependiente (survived)
train.columns.values
# Datos faltantes
missing_data = train.isnull().sum().sort_values(ascending=False)
missing_data.head()
PassengerId = test[['PassengerId']].copy()
PassengerId
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
# se define los modelos que se van a ulizar 
# se utilizan los mismos que se utilizaron en la parte superior
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict

def base_learners():
    random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
    svc_lin = SVC(kernel = 'linear', gamma = 'scale', random_state = 0)
    svc_rbf = SVC(kernel = 'rbf', random_state = 0)
    decision_tree = DecisionTreeClassifier()
    knn = KNeighborsClassifier(n_neighbors = 3) 
    logreg = LogisticRegression()

    models= {
        'SVM linear':svc_lin,
        'SVM RFV':svc_rbf,
        'KNN':knn, 
        'Logistic Regression': logreg,
        'Random Forest': random_forest,
         'Decision Tree': decision_tree
            }
    return models
from sklearn.ensemble import VotingClassifier
models = base_learners()

# Se utiliza VotingClassifier este clasificador da el resultado de prediicion prodeccion promedio basado en la prediccion de los submodelos que se establecieron en base_learners
Ensemble = VotingClassifier(estimators=list(zip(models.keys(),models.values())))
Ensemble.fit(X_train, Y_train)

scores= cross_val_score(Ensemble, X_train, Y_train, cv=10, scoring = 'accuracy')
print ('CV Ensemble Score', scores.mean())
Ensemble_Survived= pd.Series(Ensemble.predict(X_test),name= 'Survived')

predictions_ensemble = [ "0" if x < 0.5 else "1" for x in Ensemble_Survived]
predictions_ensemble

survived_predictions_ensemble = pd.DataFrame(predictions_ensemble)
survived_predictions_ensemble

submissionE = pd.DataFrame(PassengerId.join(survived_predictions_ensemble))
submissionE.columns = ['PassengerId','Survived']
submissionE.to_csv('Y_predictionEBS.csv', index=False)
submissionE
