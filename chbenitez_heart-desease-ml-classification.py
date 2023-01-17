#Importamos las librerias

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns
#Importamos los datos

df_heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
df_heart.shape
df_heart.head()
df_heart.isnull().sum()
plt.figure(figsize=(6,4))

plt.title("Cantidad de ")

sns.countplot(df_heart['target'])
plt.figure(figsize=(10,6))

plt.title("Distribucion de edad")

sns.distplot(df_heart["age"])
pd.crosstab(df_heart.sex,df_heart.target).plot(kind="bar",figsize=(8,6), color=['#459CFF','#AA1111' ])

plt.title('Frecuencia de problemas cardiacos por sexo')

plt.xlabel('Sexo (0 = Femenino, 1 = Masculino)')

plt.xticks(rotation=0)

plt.legend(["Sano", "Enfermo"])

plt.ylabel('Frecuencia')

plt.show()
plt.figure(figsize=(14,6))

sns.heatmap(df_heart.corr(),cmap='coolwarm',annot=True)
X = df_heart.drop(['target'], axis=1)

y = df_heart['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LogisticRegression

clasif_RL = LogisticRegression(random_state = 12)

#Entrenamos el modelo

clasif_RL.fit(X_train, y_train)
y_pred  = clasif_RL.predict(X_test)
from sklearn.metrics import confusion_matrix

#Funcion para graficar la matriz de confusion

def confusion(ytest,y_pred):

    names=["Neg","Pos"]

    cm=confusion_matrix(ytest,y_pred)

    f,ax=plt.subplots(figsize=(6,6))

    sns.heatmap(cm,annot=True,linewidth=.5,fmt=".0f",ax=ax,cmap="Blues", annot_kws={"size": 14})

    plt.xlabel("y_pred")

    plt.ylabel("y_true")

    ax.set_xticklabels(names)

    ax.set_yticklabels(names)

    plt.show()



    return
confusion(y_test,y_pred)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
#Funcion para graficar la curva

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='r', label="ROC - AUC={:.3f}".format(roc_auc_score(y_test, y_pred)))

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='-')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend(loc='lower right')

    plt.show()
#Dividimos los datos en FalsePositive,TruePositive y un Treshold

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

#Los utilizamos para graficar el ROC-AUC

plot_roc_curve(fpr, tpr)
#Aplicamos la metrica de la curva ROC

print("ROC AUC score:", roc_auc_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier



clasif_knn = KNeighborsClassifier()

#Entrenamos el modelo

clasif_knn.fit(X_train, y_train)
y_pred  = clasif_knn.predict(X_test)
#Utilizamos la funcion nuevamente 

confusion(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr)
print("ROC AUC score:", roc_auc_score(y_test, y_pred))
from sklearn.svm import SVC



clasif_SVM = SVC(kernel = "linear", random_state = 0)

#Entrenamos el modelo

clasif_SVM.fit(X_train, y_train)
y_pred  = clasif_SVM.predict(X_test)
confusion(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr)
print("ROC AUC score:", roc_auc_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

clasif_DT = DecisionTreeClassifier(random_state = 0)

#Entrenamos el modelo

clasif_DT.fit(X_train, y_train)
y_pred  = clasif_DT.predict(X_test)
confusion(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr)
print("ROC AUC score:", roc_auc_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



clasif_RF = RandomForestClassifier(random_state = 0)

#Entrenamos el modelo

clasif_RF.fit(X_train, y_train)
y_pred  = clasif_RF.predict(X_test)
confusion(y_test,y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plot_roc_curve(fpr, tpr)
print("ROC AUC score:", roc_auc_score(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
#Cargamos la grilla con los hiperparametros

param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
#Cargamos el modelo

clf_SVC = SVC(random_state=0)



grid_search = GridSearchCV(clf_SVC, param_grid, cv=5,

                           scoring='roc_auc', 

                           return_train_score=True,

                           n_jobs=-1)
grid_search.fit(X_train, y_train)
#Vemos los resultados del GridSearch

results = pd.DataFrame(grid_search.cv_results_)

results.head()
#Mejores parametros del modelo

grid_search.best_params_
print("El mejor score es:", grid_search.best_score_) 
#Tomamos el mejor estimador

optimised_SVC = grid_search.best_estimator_
#Tomamos un ejemplo del conjunto de test

test_predict= X_test[60:61]

test_predict
print("El paciente", list(test_predict.index),"Tiene un diagnostico:", optimised_SVC.predict(test_predict),"(Positivo: 1, Negativo: 0)")