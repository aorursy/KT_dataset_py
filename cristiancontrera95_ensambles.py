import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
df = pd.read_csv('/kaggle/input/Banco-PF-DSCOR9/data_train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/Banco-PF-DSCOR9/data_test.csv', index_col='id')
df.head(3)
df_test.head(3)
# Cambiamos el target a numeros
df.y = df.y.map({'yes': 1, 'no': 0})
df.job.value_counts()
def job_transform(df_):
    # probamos agrupar por activamente trabajando y no
    hasnt_work = ['retired', 'unemployed', 'student', 'unknown']
    df_.loc[df_.job.isin(hasnt_work), 'has_work'] = 0
    df_.loc[~df_.job.isin(hasnt_work), 'has_work'] = 1

    # df.loc[df.job.isin(hasnt_work), 'job'] = 'unemployed'

    # encodeamos las variables
    df_ = pd.concat([df_.drop(columns=['job']), pd.get_dummies(df_.job, prefix='job')], axis=1)
    return df_

df = job_transform(df)
df_test = job_transform(df_test)
df.marital.value_counts()
def marital_transform(df_):
    # como la mayoria estan casados armamos solo 2 grupos (married y otros)
    df_.marital = df_.marital.map(lambda x: 1 if x == 'married' else 0)
    return df_

df = marital_transform(df)
df_test = marital_transform(df_test)
df.education.value_counts()
df_test.education.value_counts()
def education_transform(df_):
    from sklearn.preprocessing import LabelEncoder

    df_.loc[df_.education.isin(['basic.4y', 'basic.6y', 'unknown']), 'education'] = 'basic.9y'  # porque unknown a basic.9y? no hay porque

    le = LabelEncoder()
    le.fit(['illiterate', 'basic.9y', 'high.school', 'professional.course', 'university.degree'])  # damos el orden
    df_.education = le.transform(df_.education)
    return df_

df = education_transform(df)
df_test = education_transform(df_test)
def y_las_demas_transforms(df_):

    for col in ['housing', 'loan', 'default', 'contact', 'poutcome']:
        df_.loc[df_[col] == 'unknown', col] = 'no'

        # encodeamos las variables
        df_ = pd.concat([df_.drop(columns=[col]), pd.get_dummies(df_[col], prefix=col)], axis=1)
    return df_

df = y_las_demas_transforms(df)
df_test = y_las_demas_transforms(df_test)
# 
# aca podriamos usar las funciones para fechas del modulo datetime
# from datetime import datetime as dt
# pd.to_datetime(['Wed', 'Thu', 'Mon', 'Tue', 'Fri'], format='%a')
# 
# https://stackoverflow.com/questions/62205571/pd-to-datetime-doesnt-work-with-a-format
#
df.day_of_week = df.day_of_week.str.title().map({'Sun': 1, 'Mon': 2, 'Tue': 3, 'Wed': 4, 'Thu': 5, 'Fri': 6, 'Sat': 7})
df_test.day_of_week = df_test.day_of_week.str.title().map({'Sun': 1, 'Mon': 2, 'Tue': 3, 'Wed': 4, 'Thu': 5, 'Fri': 6, 'Sat': 7})
df.month = pd.to_datetime(df.month, format='%b').dt.month
df_test.month = pd.to_datetime(df_test.month, format='%b').dt.month
df.head(3)
df_test.head(3)
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.metrics import classification_report, f1_score

sns.set()
# Separamos las features del targer
X = df.drop(columns=['y'])
y = df.y
X.shape
# Separamos los conjuntos de train y test, usamos stratify=y para que train y test tengan la misma distribucion de y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16, stratify=y)
X_train.shape, X_test.shape
# Comprobamos que tengan el mismo porcentaje de instancias de cada clase
sum(y_train)/len(y_train), sum(y_test)/len(y_test)
# Creamos un objecto KFold que vamos a usar para cross_validation
kf = KFold(n_splits=3, shuffle=True)
from sklearn.metrics import roc_auc_score, roc_curve
kf  # vamos a usar el mismo KFold
param_grid = {
    'max_depth' : [2, 5, 10],
    'max_features' : ['sqrt', 'log2', X_test.shape[1]//2],
    'n_estimators' :[10, 20, 50],
    'class_weight': [{0:1, 1:5}, {0:1, 1:2}],
}
rf = RandomForestClassifier(min_samples_split=10, oob_score=True, random_state=16)
gs_rf = GridSearchCV(rf, param_grid=param_grid, scoring='f1', cv=kf, verbose=1, n_jobs=-1, return_train_score=True)
%%time
gs_rf.fit(X_train, y_train)
gs_rf.best_estimator_
print(classification_report(y_test, gs_rf.predict(X_test)))
gs_rf.best_estimator_.oob_score_
def plot_features_importance(estimator, column_names):
    # Obtenemos el orden por importance 
    indices = np.argsort(estimator.feature_importances_)

    # ploteamos
    plt.figure(figsize=(16, 10))
    sns.barplot(estimator.feature_importances_[indices], column_names[indices])
    plt.title('Feature Importances')
    plt.show();
plot_features_importance(gs_rf.best_estimator_, X_train.columns)
# Definimos una funcion para plotear curva roc ya que la vamos a volver a usar

def plot_roc_curve(estimator, X, y, name=''):
    #Obtaining the ROC score
    roc_auc = roc_auc_score(y, estimator.predict(X))
    #Obtaining false and true positives & thresholds
    fpr, tpr, thresholds = roc_curve(y, estimator.predict_proba(X)[:,1])
    
    #Plotting the curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='roc test')

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve {name} (area = {roc_auc:0.03f})')
    plt.legend()
    plt.show();
    
plot_roc_curve(gs_rf, X_test, y_test, name='Random Forest')
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
xline = X_train['emp.var.rate']
yline = X_train['euribor3m']
zline = X_train['nr.employed']

ax.scatter3D(xline, yline, zline, c=y_train)  # c=y_train para diferencias las clases
param_grid = {
    'n_estimators': [10, 40, 100],
    'learning_rate': np.linspace(1e-3, 1, 5)
}
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm='SAMME.R', random_state=16)
gs_ada = GridSearchCV(ada, param_grid=param_grid, scoring='f1', cv=kf, verbose=1, n_jobs=-1, return_train_score=True)
%%time
# Tener en cuenta que por la cantidad de fit de GridSeach (la combianacion de los hiperametros) estan los fit internos de AdaBoost
gs_ada.fit(X_train, y_train)
gs_ada.best_estimator_
print(classification_report(y_test, gs_ada.predict(X_test)))
f1_test = []
f1_train = []

# Calculamos el accuracy sobre el test set
for prediccion_test in gs_ada.best_estimator_.staged_predict(X_test):
    f1_test.append(f1_score(y_test, prediccion_test))

for prediccion_train in gs_ada.best_estimator_.staged_predict(X_train):    
    f1_train.append(f1_score(y_train, prediccion_train))

plt.figure(figsize=(12, 6))

# ploteamos las lineas de train y test
plt.plot(range(1, len(f1_train) + 1), f1_train, label = 'Train', color="b")
plt.plot(range(1, len(f1_test) + 1), f1_test, label = 'Test', color="r")

plt.legend()
plt.title(f'F1-Score {f1_score(gs_ada.predict(X_test), y_test):.2f}')
plt.ylabel('f1')
plt.ylim([0.2, 0.7])
plt.xlabel('Cantidad de estimadores')
plt.show();
# !pip install xgboost
import xgboost as xgb
# dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns, nthread=-1)
# dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns, nthread=-1)
# Definimos nuestra metrica f1 personalizada para xgboost
# 1 - f1_score es porque queremos que 

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', gamma=0.01, min_child_weight=3, verbosity=1, random_state=16)
sum(y==0)/sum(y==1)
# ponemos varios parametros al azar realmente leyendo solo un poco la documentacion https://xgboost.readthedocs.io/en/latest/parameter.html

param_grid = [
    { # booster gbtree
    'booster': ['gbtree'],
    'max_depth': [2, 4, 5],
    'learning_rate': np.linspace(1e-5, 1, 4),
    'n_estimators': [100, 200],
    'scale_pos_weight': np.arange(1, 15, 4)  # recomendado: sum(y_train==0)/sum(y_train==1) 
    }
]
gs_xgb = GridSearchCV(xgb_clf, param_grid=param_grid, scoring='f1', cv=kf, verbose=1, n_jobs=-1, return_train_score=True)
%%time
gs_xgb.fit(X_train, y_train,
       eval_set=[(X_train, y_train), (X_test, y_test)],
       eval_metric=f1_eval,
       early_stopping_rounds=20,
       verbose=True)
gs_xgb.best_estimator_
print(classification_report(y_test, gs_xgb.predict(X_test)))
plot_features_importance(gs_xgb.best_estimator_, X_train.columns)
# pip install catboost
# pip install ipywidgets  # libreria para plot en jupyter
from catboost import CatBoostClassifier
cat_boost = CatBoostClassifier( random_seed=16)  # tiene mas hiperparametros que ..
grid = {
    'n_estimators': [60, 100, 150],
    'learning_rate': [0.03, 0.1],
    'depth': [4, 6, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}
%%time
grid_search_result = cat_boost.grid_search(grid, 
                                           X=X_train,
                                           y=y_train, 
                                           plot=True,
                                           cv=kf)
print(classification_report(y_test, cat_boost.predict(X_test)))
# !pip install lightgbm
from lightgbm import LGBMClassifier
clf_lgbm = LGBMClassifier(objective='binary', random_state=16)
param_grid = {
    'n_estimators': [60,100,150],
    'boosting_type': ['gbdt', 'rf'],
    'max_depth': [2,5,10],
    'learning_rate': np.linspace(1e-5, 1, 5),
}
gs_lgbm = GridSearchCV(clf_lgbm, param_grid=param_grid, scoring='f1', cv=kf, verbose=1, n_jobs=-1, return_train_score=True)
%%time
gs_lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='binary')
print(classification_report(y_test, gs_lgbm.predict(X_test)))
from sklearn.svm import SVC
svm = SVC(probability=True, random_state=16)
grid = [
#     {
#         'kernel': ['linear'],
#         'C': np.linspace(0.01, 2, 5),
#     }, 
    {
        'kernel': ['rbf', 'poly'],
        'C': np.linspace(1e-2, 1, 5),
        'degree': [2,3,5],
#         'gamma': ['scale', 'auto'],
#         'coef0': np.arange(0, 10, 3),
    }
]
gs_svm = GridSearchCV(svm, param_grid=grid, scoring='f1', cv=kf, verbose=1, n_jobs=-1, return_train_score=True)
%%time
gs_svm.fit(X_train, y_train)
print(classification_report(y_test, gs_svm.predict(X_test)))
# vamos a medir el resultado de f1 para distintos pesos entre xgboost y svm
for p in np.linspace(0, 1, 6):
    predict_probs = gs_svm.predict_proba(X_test)[:,1] * p + gs_xgb.predict_proba(X_test)[:,1] * (1-p)

    predict = np.where(predict_probs >= 0.5, 1, 0)

    print(f'svm * {p:.1f} + xgboost * {1-p:.1f}: {f1_score(y_test, predict, average="macro"):.4f}')
# vamos a medir el resultado de f1 para distintos pesos entre rf y svm
for p in np.linspace(0, 1, 6):
    predict_probs = gs_rf.predict_proba(X_test)[:,1] * p + gs_xgb.predict_proba(X_test)[:,1] * (1-p)

    predict = np.where(predict_probs >= 0.5, 1, 0)

    print(f'rf * {p:.1f} + xgboost * {1-p:.1f}: {f1_score(y_test, predict, average="macro"):.4f}')
# Vamos con estos pesos

predict_probs = gs_svm.predict_proba(df_test)[:,1] * 0.2 + gs_xgb.predict_proba(df_test)[:,1] * 0.8

predict = np.where(predict_probs >= 0.5, 1, 0)
sum(predict), sum(predict)/len(predict) # Total de 1
pd.Series(predict, name='y').to_csv('sample_submit.csv', index_label='id')
!head sample_submit.csv
