# Importar las librerías necesarias

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split # Para separar Train y Test

from sklearn import metrics # Para medir la efectividad de los modelos

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



import matplotlib.pyplot as plt 

from IPython.display import Image

import pydotplus # Si no lo tienen instalado: conda install -c conda-forge pydotplus
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#Imputamos los nulos que figuran como "vacios"

data['TotalCharges'] = data['TotalCharges'].replace(' ',-1).astype(float)
# Seleccionamos las variables categóricas

cat_vars = ['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService',

           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

           'PaymentMethod']
# Iteramos sobre cada variable creando su dummie         

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(data[var], prefix=var)

    data1=data.join(cat_list)

    data=data1
# Descartamos las variables originales

data = data.drop(cat_vars, axis = 1)
# El target también los convertimos en una variable numérica dummie

data['target'] = np.where(data.Churn == 'Yes',1,0)
# Eliminamos la variable Target y el ID de cliente que no arroja información (realmente no tiene información?)

data = data.drop(['Churn', 'customerID'], axis = 1)
data.head()
# Separamos la base en las columnas Independientes y la Dependiente (X e Y)

X, y = data.drop(data.columns[-1], axis=1), data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Tamaño de Base:", data.shape)

print("Tamaño de Muestra de Entrenamiento:", X_train.shape)

print("Tamaño de Muestra de Testeo", X_test.shape)

print("Tamaño del Target de Entrenamiento:", y_train.shape)

print("Tamaño del Target de Testeo", y_test.shape)
#Guardo un objeto con las metricas de mis modelos

metricas = {}
# Entreno un Arbol de Decision

dtree= DecisionTreeClassifier(max_depth = 3)

dtree.fit(X_train,y_train)
# Predigo sobre la base de Validación

y_pred = dtree.predict(X_test)



# Me guardo las Metricas que necesito para graficar

auc = metrics.roc_auc_score(np.asarray(y_test), y_pred)

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test), y_pred)

metricas ['decisionTree'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
# Graficamos el Arbol para entenderlo

dot_data = tree.export_graphviz(dtree, out_file=None, 

                                feature_names= X.columns,

                                 class_names= ['No','Si'])



graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
# Realizamos una tabla cruzada para ver la efectividad del resultado

pd.crosstab(np.asarray(y_test), y_pred)
#Entreno una Regresión Logistica

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
# Predigo sobre la base de Validación

y_pred = logreg.predict(X_test)



# Me guardo las Metricas que necesito para graficar

auc = metrics.roc_auc_score(np.asarray(y_test), y_pred)

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test), y_pred)

metricas['logisticRegresion'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
# Realizamos una tabla cruzada para ver la efectividad del resultado

pd.crosstab(np.asarray(y_test), y_pred)
# Calculo el Accuracy 

# ¿Qué porcentaje de predicciones fue correcta? 

# (0 & 0) + (1 & 1) / Total = 

metrics.accuracy_score(np.asarray(y_test), y_pred)
# Calculo el Error Medio Absoluto 

# ¿Qué porcentaje de predicciones fue incorrecta?

# (0 & 1) + (1 & 0) / Total = 

metrics.mean_absolute_error(np.asarray(y_test), y_pred)
# Calculo el Recall 

# ¿Qué porcentaje de casos positivos fueron capturados? 

# (1 & 1) / (1 & 0) + (1 & 1) = 

metrics.recall_score(np.asarray(y_test), y_pred)
# Calculo de la Precisión

# ¿Qué porcentaje de predicciones positivos fueron correctas? 

# (1 & 1) / (0 & 1) + (1 & 1) = 

metrics.precision_score(np.asarray(y_test), y_pred)
#Entreno un Random Forest

RF = RandomForestClassifier()

RF.fit(X_train, y_train)
# Predigo sobre la base de Validación

y_pred = RF.predict(X_test)



# Me guardo las Metricas que necesito para graficar

auc = metrics.roc_auc_score(np.asarray(y_test), y_pred)

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test), y_pred)

metricas['RandomForest'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
# Entreno un Modelo de AdaBoost

AdaB = AdaBoostClassifier()

AdaB.fit(X_train,y_train)
# Predigo sobre la base de Validación

y_pred = AdaB.predict(X_test)



# Me guardo las Metricas que necesito para graficar

auc = metrics.roc_auc_score(np.asarray(y_test), y_pred)

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test), y_pred)

metricas ['AdaBoost'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
# Entreno un Modelo de Gradient Boosting

GBM = GradientBoostingClassifier()

GBM.fit(X_train,y_train)



# Me guardo las Metricas que necesito para graficar

y_pred = GBM.predict(X_test)

auc = metrics.roc_auc_score(np.asarray(y_test), y_pred)

fpr, tpr, thresholds = metrics.roc_curve(np.asarray(y_test), y_pred)

metricas ['GradientBoosting'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
# Grafico la Curva ROC con los valores de mis modelos

for modelName in metricas:

    label = 'ROC curve for {0}:'.format(modelName)

    plt.plot(metricas[modelName]['fpr'], metricas[modelName]['tpr'], label=label+'AUC={0:0.2f}'.format(metricas[modelName]['auc']))

    plt.xlabel('1-Specificity')

    plt.ylabel('Sensitivity')

    plt.ylim([0.0, 1.0])

    plt.xlim([0.0, 1.0])

    plt.plot([0, 1], [0, 1], 'k--')



plt.grid(True)

plt.title('ROC')

plt.legend(loc="lower left")

plt.show()
