import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import gc
train = pd.read_csv('D:/Cursos Data Science/Kaggle/train.csv')
test  = pd.read_csv("D:/Cursos Data Science/Kaggle/test.csv")
train.head()
test.head()
print(train.shape)
print(test.shape)
train.describe()
train.dtypes.to_csv('D:/Cursos Data Science/Kaggle/variables_train.csv')
variables_mixtas = (8,9,10,11,12,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,38,39,40,41,42,43,44,58,59,60,61,62,63,66,67,68,69,70,71,72,73,75,76,77,
78,79,80,83,84,114,156,157,158,159,166,167,168,169,176,177,178,179,188,189,190,191,196,202,204,207,213,214,216,217,222,225,228,229,231,235,238,239,
244,266,283,305,404,427,428,454,466,467,493,840)
print(len(variables_mixtas))
# Recolección de variables mixtas
cols = []
variables_mixtas_cols = []
for i in range(1,1935):
    if i not in variables_mixtas:
        cols.append(i)
    else:
        variables_mixtas_cols.append(i)
# Lista de variables mixtas
cols.sort()
variables_mixtas_cols.sort()
cols = [str(n).zfill(4) for n in cols]
cols = ['VAR_' + n for n in cols] 
cols.append('target')
cols.insert(0,'ID')

variables_mixtas_cols = [str(n).zfill(4) for n in variables_mixtas_cols]
variables_mixtas_cols = ['VAR_' + n for n in variables_mixtas_cols] 
print(train.shape)
print(test.shape)
# Eliminar variables mixtas en la data train
train.drop(variables_mixtas_cols, axis=1, inplace=True)
# Eliminar variables mixtas en la data test
test.drop(variables_mixtas_cols, axis=1, inplace=True)
print(train.shape)
print(test.shape)
# Verificar el porcentaje de nulos por variable
columnas_nulas = train.isnull().sum()/len(train)
columnas_nulas.to_csv('D:/Cursos Data Science/Kaggle/null_values.csv')
# Elimino los que tienen el valor de nulo superior al 50% del total de datos para la data train
train.drop(['VAR_0074','VAR_0205','VAR_0206','VAR_0208','VAR_0209','VAR_0210','VAR_0211','VAR_0226', 'VAR_0230', 'VAR_0232', 'VAR_0236'], axis=1,inplace=True)
# Elimino los que tienen el valor de nulo superior al 50% del total de datos para la data test
test.drop(['VAR_0074','VAR_0205','VAR_0206','VAR_0208','VAR_0209','VAR_0210','VAR_0211','VAR_0226', 'VAR_0230', 'VAR_0232', 'VAR_0236'], axis=1,inplace=True)
print(train.shape)
print(test.shape)
train.describe()
# Variables que tengo para modelar
features = list(train.columns)
train_copia = train
# features
#características numéricas
features_num = list(train.describe())
# features_num
#características categóricas
features_cat = list(train_copia[features].drop(features_num, axis=1).columns)
features_cat
# Ploteamos el comportamiento de las variables categóricas para ver qué aportan
#     var = train.groupby(feature)[feature].count().sort_values(ascending = False)
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.set_xlabel(feature)
#     ax1.set_ylabel('Cantidad')
#     ax1.set_title(feature)
#     var.plot(kind='bar')
#     plt.show()
# Definimos una función para convertir un dataframe de valores categóricos a números
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def conversion(dataframe):
    for col in dataframe:
        nans = dataframe[col].isnull().sum()
        
        if not np.isreal(dataframe[col][0]):
            if nans > 0:
                dataframe[col] = dataframe[col].fillna('Void')            
            dataframe[col] = dataframe[col].astype(str)    
            le.fit(dataframe[col])
            dataframe[col] = le.transform(dataframe[col])
        else:
            if nans > 0:
                dataframe[col] = dataframe[col].fillna(0)
    
    return dataframe    
train = conversion(train)
test = conversion(test)
train.head()
test.head()
print(train.shape)
print(test.shape)
# Vemos la cantidad de ceros como target
no = train[train['target'] == 0]
no.head()
# Vemos la cantidad de unos como target
si = train[train['target'] == 1]
si.head()
print(len(no))
print(len(si))
# Nos damos cuenta que la proporción es de 3.3 a 1
print(round(len(no)/len(train),4))
print(round(len(si)/len(train),4))
# En lugar de usar el undersampling que más abajo detallo, utilizo solo una muestra de 38000 por temas de costo computacional
no_sub_muestra = no.sample(n=38000)
print(len(no_sub_muestra))
print(len(si))
# Creo un dataframe nuevo que es la unión de la muestra de 38k ceros y todos los unos con los que contaba
train_2 = pd.concat([no_sub_muestra, si], axis=0)
len(train_2)
select = [x for x in train_2.columns if x != 'target']
X = train_2.loc[:, select]
y = train_2['target']
# Verifico que ambos cuenten con la misma cantidad de datos
print(X.shape)
print(y.shape)
# Lo óptimo es usar un modelo con cross validation, pero por un tema de costo computacional no lo he usado. De igual forma lo detallo en cada modelo utilizado.
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.70, random_state=999)
print(len(X_train), len(y_train), len(X_test), len(y_test))
X_train = X
y_train = y
len(X), len(y)
# import random
# def UnderSampling(X, y, target_percentage, seed):
#     # Assuming minority class is the positive
#     n_samples = y.shape[0]
#     n_samples_0 = (y == 0).sum()
#     n_samples_1 = (y == 1).sum()

#     n_samples_0_new =  n_samples_1 / target_percentage - n_samples_1
#     n_samples_0_new_per = n_samples_0_new / n_samples_0

#     filter_ = y == 0

#     np.random.seed(seed)
#     rand_1 = np.random.binomial(n=1, p=n_samples_0_new_per, size=n_samples)
    
#     filter_ = filter_ & rand_1
#     filter_ = filter_ | (y == 1)
#     filter_ = filter_.astype(bool)
    
#     return X[filter_], y[filter_]
# from time import time
# time_star = time()

# X_u, y_u = UnderSampling(X_train, y_train, 0.45, 103)

# time_end = time()
# print ("Time: ", np.round((time_end-time_star)/60,2), " minutes")
# X_u.shape
# y_u.shape
seeds = np.arange(0,501,50)
seeds
from time import time
time_star = time()

temp = pd.DataFrame({'atributo':list(X.columns)})
from sklearn.ensemble import RandomForestClassifier
for seed in seeds:
    clf = RandomForestClassifier(random_state=seed)
    clf = clf.fit(X, y)
    semilla = 'semilla_' + str(seed)
    temp[semilla]=clf.feature_importances_
temp['importancia'] = temp.iloc[:,1:].apply(np.mean, axis=1)

time_end = time()
print ("Time: ", np.round((time_end-time_star)/60,2), " minutes")
ranking_features = temp[['atributo','importancia']].sort_values('importancia', ascending = False).reset_index(drop = True)
ranking_features.head(250)
variables_elegidas = ranking_features.iloc[:, 0].head(250)
X[variables_elegidas].head()
X_train = X[variables_elegidas]
X_test = X_test[variables_elegidas]
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
mean_accuracy_scores = []
clf = logreg.fit(X_train, y_train)
scores = cross_val_score(clf, X_train, y_train, cv=10)
mean_accuracy_scores.append(np.mean(scores))
print (mean_accuracy_scores)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred)
print(cm)
print("accuracy = " + str(accuracy))
print("roc_auc = " + str(roc_auc))
print(classification_report(y_test, y_pred))
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# # Entrenar la data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)

# Atención: comento el método con cross validation porque el costo computacional es muy alto.

# model = XGBClassifier()
# kfold = KFold(n_splits=10, random_state=7)
# results = cross_val_score(model, X_train, y_train, cv=kfold)
# print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Predicciones sobre la data de test
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# Evaluamos la predicción
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
def metricas(objetivo, prediccion):
    matriz_conf = confusion_matrix(objetivo, prediccion)
    score = accuracy_score(objetivo, prediccion)
    reporte = classification_report(objetivo, prediccion)
    metricas = [matriz_conf, score, reporte]
    return(metricas)
metricas = metricas(y_test, predictions)
[print(i) for i in metricas]
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicción del conjunto de prueba
y_pred = classifier.predict(X_test)

# Matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
roc_auc = metrics.roc_auc_score(y_test, y_pred)
print(cm)
print("accuracy = " + str(accuracy))
print("roc_auc = " + str(roc_auc))
# Como demora demasiado el SVM con cross-validation, he comentado el modelo con validación cruzada y he dejado este simple.
from sklearn.svm import SVC
clf = SVC(kernel='linear').fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("Accuracy on testing set:")
print (clf.score(X_test, y_test))
# Modelo SVM con cross-validation que demora mucho por el coste máquina

# from sklearn.svm import SVC
# mean_accuracy_scores = []
# clf = SVC(kernel='linear').fit(X_train, y_train)
# scores = cross_val_score(clf, X_train, y_train, cv=10)
# mean_accuracy_scores.append(np.mean(scores))
# print (mean_accuracy_scores)
# y_pred = clf.predict(X_test)
test_2 = test.sample(n=38000)
print(test_2.shape)
print(X_train.shape)
X_test_real = test_2
print(X_test_real.shape)
y_pred = model.predict(X_test_real)
predictions = [round(value) for value in y_pred]
resultados = X_test_real
resultados['predictions'] = predictions
resultados.head(25)
resultados.to_csv('D:/Cursos Data Science/Kaggle/resultado_data_test.csv')