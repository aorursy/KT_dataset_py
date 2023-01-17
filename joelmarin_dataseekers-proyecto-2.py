import sklearn
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('/kaggle/input/trainred/Trainred.csv')
df.shape
df ##Importamos el set de datos preprocesado
X=np.array(df[['village','age','children','edu','cons_social','cons_other',
      'ed_expenses_perkid','durable_investment','nondurable_investment','ent_farmprofit']])
X = preprocessing.StandardScaler().fit(X).transform(X)
y=np.array(df['depressed']) 
## Se crea un numpy array con las features que se seleccionaron con el RFE y se normalizan los datos
model = RandomForestClassifier(n_estimators = 128, random_state = 42)
model.fit(X, y)
scores = cross_val_score(model, X, y, cv=10)
##Se construye y ajusta el modelo con 128 árboles. 
##Además, se consigue los scores para la validación cruzada con 10 K folds. 
print('Cross validated Scores',scores)
pred = cross_val_predict(model, X, y, cv=10)
print('Mean Absolute Error:', metrics.mean_absolute_error(y, pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y, pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, pred)))
print (classification_report(y, pred))
##Obtenemos las métricas que se tienen del modelo. 
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función muestra y dibuja la matriz de confusión.
    La normalización se puede aplicar estableciendo el valor `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
print(confusion_matrix(y, pred, labels=[1,0]))

## Este código inicia la construcción de la matriz de confusión para el modelo escogido. 
cnf_matrix = confusion_matrix(y, pred, labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Not Depressed','Depressed'],normalize= False,  title='Matriz de confusión')
##Se construye la matriz de confusión