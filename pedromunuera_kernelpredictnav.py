# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline



#Clasificadores

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import xgboost as xgb





#Spoliler: la metrica de accuracy se va a quedar corta

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



#Busqueda en grid de hiperparametros

from sklearn.model_selection import GridSearchCV



#graficas

import seaborn as sn

import matplotlib.pyplot as plt



#Sampling

from imblearn.over_sampling import SMOTE
#Cargamos el dataset

dataset=pd.read_csv('../input/dataset_beta_an.csv');

#Eliminamos algunas columnas que habia en el dataset anterior

#El hash de la ip es una llamada al overfitting

#dataset.drop(['ip_hash','fecha','lang','country'], axis=1, inplace=True)



#Separamos la columna target

X, y = dataset.iloc[:, 0:-1].values, dataset.iloc[:, -1].values

#Creamos 2 conjuntos de datos los de entrenamiento y los de validación

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



#Vamos a generar muchas matrices de error

def showConfusionMat(pipe, X_test,y_test,title=''):

    plt.figure()

    plt.title(title)

    y_pred = pipe.predict(X_test)

    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

    print(title)

    print(confmat)

    df_cm = pd.DataFrame(confmat, ['F','T'],['F','T'])

    sn.set(font_scale=1.4)

    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})

    

#Para sacar graficas en 2D

def show2Dscatter(X,y,title=''):

    pipeline_pca = Pipeline([('scl', StandardScaler()),

    ('pca', PCA(n_components=2)),])



    X_2d=pipeline_pca.fit_transform(X)

    plt.figure()

    plt.title(title)

    plt.scatter(X_2d[:,0],X_2d[:,1],c=y)

    plt.show()

    

def show2DscatterNorm(X,y,title=''):

    pipeline_pca = Pipeline([('scl', StandardScaler()),

    ('pca', PCA(n_components=2)),])



    X_2d=pipeline_pca.fit_transform(X)

    plt.figure()

    plt.title(title)

    plt.scatter(X_2d[:,0],X_2d[:,1],c=y)

    plt.show()

    

def showResults(pipe, X_test,y_test,title=''):    

    print('Mejor puntuacion en sensibilidad: ')

    print(pipe.best_score_)

    print(pipe.best_params_)

    print(accuracy_score(y_test, pipe.predict(X_test_l)))

    showConfusionMat(pipe,X_test,y_test,'RForest con Sensibilidad')

    print(classification_report(y_test, pipe.predict(X_test)))
dataset.head(5)
#Empezamos con una pipeline para facilitar el trabajo

pipeline_lr = Pipeline([('scl', StandardScaler()),

('pca', PCA(n_components=2)),

('clf', LogisticRegression(solver='lbfgs',random_state=1))])



pipeline_lr.fit(X_train, y_train)



print('LogisticRegression')

print('Resultado: %.3f' % pipeline_lr.score(X_test, y_test))



#Pileline del SVM

pipeline_svm = Pipeline([('scl', StandardScaler()),

('clf', SVC(kernel='linear', C=0.05,random_state=1))])



print('SVC')

pipeline_svm.fit(X_train, y_train)

print('Resultado: %.3f' % pipeline_svm.score(X_test, y_test))

print('Datos Entrenamiento:')

print(y_train)

print('Datos Test:')

print(y_test)

print('Pocos 1s')

#Mostramos los datos de los estimadores

showConfusionMat(pipeline_lr, X_test,y_test,'Logit')

print(classification_report(y_test, pipeline_lr.predict(X_test)))

showConfusionMat(pipeline_svm,X_test,y_test,'SVM')

print(classification_report(y_test, pipeline_svm.predict(X_test)))
target_count = dataset.target.value_counts()

print('Class 0 Negativa:', target_count[0])

print('Class 1: Positiva', target_count[1])

print('Proporcion:', round(target_count[0] / target_count[1], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)');
#Resampleo de los 1s 

sm = SMOTE(random_state=1, ratio = 1.0)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



pipeline_lr.fit(X_train_res, y_train_res)

print('Resultado accu: %.3f' % pipeline_lr.score(X_test, y_test))



showConfusionMat(pipeline_lr, X_test,y_test,'Logit con SMOTE')

print(classification_report(y_test, pipeline_lr.predict(X_test)))



#Prueba dejando que el modelo haga el balanceo en lugar de hacerlo nosotros

pipeline_lr_b = Pipeline([('scl', StandardScaler()),

('pca', PCA(n_components=2)),

('clf', LogisticRegression(solver='lbfgs',class_weight='balanced',random_state=1))])



#Ahora dejamos que sea el modelo el que balancee

pipeline_lr_b.fit(X_train, y_train)

print('Resultado accu: %.3f' % pipeline_lr_b.score(X_test, y_test))



showConfusionMat(pipeline_lr_b, X_test,y_test,'Logit con Autobalanceo')

print(classification_report(y_test, pipeline_lr_b.predict(X_test)))







rmfc=RandomForestClassifier(n_estimators=100)

rmfc=rmfc.fit(X_train_res,y_train_res)



print('Resultado: %.3f' % rmfc.score(X_test, y_test))

showConfusionMat(rmfc,X_test,y_test,'Radom Forest')

print(classification_report(y_test, rmfc.predict(X_test)))

#Aqui podriamos meter mas modelos para iterar con sus propios hiperparametros

#Por ejemplo las pipelines del comienzo

#Pero creo que la idea se entiende, y no es plan de dejar sin tiempo de CPU el kernel

#Lo ideal sería cargar el GridSearch con 4 o 5 horas de trabajo,lanzar en la nube y mientras a otra cosa

pipe= Pipeline([('rfc', RandomForestClassifier())])





#param_grid = [{'rfc__n_estimators': [50,75,90,100,110,150,200,300,400]}] Vamos afinando

param_grid = [{'rfc__n_estimators': [90,95,100,105,110,115,120,125]}] 

gs = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring='recall',cv=5,n_jobs=-1)

gs = gs.fit(X_train_res, y_train_res)

print('Mejor puntuacion en sensibilidad')

print(gs.best_score_)

print(gs.best_params_)

showConfusionMat(gs,X_test,y_test,'Random Forest')

print(classification_report(y_test, gs.predict(X_test)))


gbm = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.1)

gbm.fit(X_train_res, y_train_res)

print('Resultado precision: %.3f' % gbm.score(X_test, y_test))

showConfusionMat(gbm,X_test,y_test,'XGBoost')

print(classification_report(y_test, gbm.predict(X_test)))

print(list(dataset.columns.values[0:-1]))



print(list(gbm.feature_importances_))
pipe= Pipeline([('gbm', xgb.XGBClassifier())])





param_grid = [{'gbm__max_depth': [4,5,6],'gbm__n_estimators': [180,200,300],'gbm__learning_rate': [0.15,0.1,0.5]}]

gs = GridSearchCV(estimator=pipe,param_grid=param_grid,scoring='accuracy',cv=5,n_jobs=-1)

gs = gs.fit(X_train_res, y_train_res)

print(gs.best_score_)

print(gs.best_params_)

showConfusionMat(gs,X_test,y_test,'XGBoost con Accuracy')

print(classification_report(y_test, gs.predict(X_test)))





pipe_gbm = Pipeline([('gbm', xgb.XGBClassifier())])



param_grid = [{'gbm__max_depth': [3,4,5],'gbm__n_estimators': [250,300,350],'gbm__learning_rate': [0.2,0.1,0.5]}]

gs2 = GridSearchCV(estimator=pipe_gbm,param_grid=param_grid,scoring='f1',cv=5,n_jobs=-1)

gs2 = gs2.fit(X_train_res, y_train_res)

print('Mejor F1:')

print(gs2.best_score_)

print(gs2.best_params_)

showConfusionMat(gs2,X_test,y_test,'XGBoost con F1')
dataset_large=pd.read_csv('../input/dataset_beta_plus.csv');

#Codificamos categorias de los datos



dataset_large['hora'] = dataset_large['hora'].astype('category')

dataset_large['hora']=dataset_large['hora'].cat.codes



dataset_large['country_id'] = dataset_large['country_id'].astype('category')

dataset_large['country_id']=dataset_large['country_id'].cat.codes



dataset_large['lang_id'] = dataset_large['lang_id'].astype('category')

dataset_large['lang_id']=dataset_large['lang_id'].cat.codes



#Todo a 0`s lo borramos

dataset_large.drop(['has_post'], axis=1, inplace=True)

X_l, y_l = dataset_large.iloc[:, 0:-1].values, dataset_large.iloc[:, -1].values

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.3, random_state=0)



sm = SMOTE(random_state=1, ratio = 1.0)

X_train_res_l, y_train_res_l = sm.fit_sample(X_train_l, y_train_l)
pipe_l= Pipeline([('rfc', RandomForestClassifier())])

param_grid = [{'rfc__n_estimators': [50,75,90,100,110,150,200,300,400]}]



gs = GridSearchCV(estimator=pipe_l,param_grid=param_grid,scoring='recall',cv=5)

gs = gs.fit(X_train_res_l, y_train_res_l)



print('Mejor puntuacion en sensibilidad: ')

print(gs.best_score_)

print(gs.best_params_)

print(accuracy_score(y_test_l, gs.predict(X_test_l)))

showConfusionMat(gs,X_test_l,y_test_l,'RForest con Sensibilidad')

print(classification_report(y_test_l, gs.predict(X_test_l)))
#Grafica sin hora

show2Dscatter(X_test,y_test,'Dataset Normal')

#Grafica con hora

show2Dscatter(X_test_l,y_test_l,'Dataset Plus')

pipe_gbm = Pipeline([('gbm', xgb.XGBClassifier())])



param_grid = [{'gbm__max_depth': [4,5,6],'gbm__n_estimators': [300,400,500,600],

               'gbm__learning_rate': [0.5,0.2,0.1,0.02]}]

gs = GridSearchCV(estimator=pipe_gbm,param_grid=param_grid,scoring='f1',cv=5,n_jobs=-1)

gs = gs.fit(X_train_res_l, y_train_res_l)

print('Mejor F1:')

print(gs.best_score_)

print(gs.best_params_)

showResults(gs,X_test_l,y_test_l,'XGBoost con F1')