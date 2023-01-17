import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
#IMPORTANDO DATAFRAMES
datasuvived = pd.read_csv('../input/titanic/gender_submission.csv')
datatest = pd.read_csv('../input/titanic/test.csv')
datatrain = pd.read_csv('../input/titanic/train.csv')
SEED = 5
np.random.seed(SEED)
def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else:
        return 0
datatrain['Sex_binario'] = datatrain['Sex'].map(transformar_sexo)
datatest['Sex_binario'] = datatest['Sex'].map(transformar_sexo)
def is_alone(x):
    if  (x['SibSp'] + x['Parch'])  > 0:
        return 0
    else:
        return 1

datatrain['Is_alone'] = datatrain.apply(is_alone, axis = 1)
datatest['Is_alone'] = datatest.apply(is_alone, axis = 1)
#PREVISÕES USANDO DUMMIES
dados = ['Sex_binario', 'Age', 'Pclass', 'Fare','Is_alone','SibSp','Parch','Embarked']
x_treino = datatrain[dados].fillna(-1)
y_treino = datatrain['Survived']

x_teste = datatest[dados].fillna(-1)
y_teste = datasuvived['Survived']

treino_x = datatrain[dados].fillna(-1)
teste_x = datatest[dados].fillna(-1)

X_testeDummies = pd.get_dummies(teste_x).astype(int)
Xdummies = pd.get_dummies(treino_x).astype(int)
Ydummies = y_treino

Xdummies = Xdummies.drop('Embarked_-1', axis=1)

X_teste = X_testeDummies.values
X = Xdummies.values
Y = Ydummies.values

#MODELO LINEAR SVC
modelo = LinearSVC(random_state=0)
modelo.fit(X,Y)
previsoes = modelo.predict(X_teste)

#MODELO RANDOM FLOREST
modelorfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
modelorfc.fit(X,Y)
previsoes2 = modelorfc.predict(X_teste)

acuracia = accuracy_score(y_teste, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

acuracia2 = accuracy_score(y_teste, previsoes2) * 100
print("A acurácia foi %.2f%%" % acuracia2)
#PREVISÕES - MODELO LINEAR SVC
dados2 = ['Sex_binario', 'Age', 'Pclass', 'Fare','Is_alone','SibSp','Parch']
x_treino = datatrain[dados2].fillna(-1)
y_treino = datatrain['Survived']

x_teste = datatest[dados2].fillna(-1)
y_teste = datasuvived['Survived']

modelo = LinearSVC(random_state=0)
modelo.fit(x_treino, y_treino)
previsoes = modelo.predict(x_teste)

acuracia = accuracy_score(y_teste, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)



#PREVISÕES - MODELO RANDOM FLOREST
dados2 = ['Sex_binario', 'Age', 'Pclass', 'Fare','Is_alone','SibSp','Parch']
x_treino = datatrain[dados2].fillna(-1)
y_treino = datatrain['Survived']

x_teste = datatest[dados2].fillna(-1)
y_teste = datasuvived['Survived']

modelorfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
modelorfc.fit(x_treino, y_treino)
previsoes2 = modelorfc.predict(x_teste)

acuracia2 = accuracy_score(y_teste, previsoes2) * 100
print("A acurácia foi %.2f%%" % acuracia2)
#BASELINE
previsoes_de_base = np.ones(418)
acuracia = accuracy_score(y_teste, previsoes_de_base) * 100
print("A acurácia do algoritmo de baseline foi %.2f%%" % acuracia)
# PREVISÕES USANDO XGBOOTS
dados2 = ['Sex_binario', 'Age', 'Pclass', 'Fare','Is_alone','SibSp','Parch']
x_treino = datatrain[dados2].fillna(-1)
y_treino = datatrain['Survived']

x_teste = datatest[dados2].fillna(-1)
y_teste = datasuvived['Survived']

data_dmatrix = xgb.DMatrix(data=X,label=Y)
xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.2, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)

xg_reg.fit(x_treino,y_treino)

preds = xg_reg.predict(x_teste)
pred = [round(value) for value in preds]
accuracy = accuracy_score(y_teste, pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

sub = pd.Series(preds, index=datatest['PassengerId'], name='Survived')
sub.to_csv("Titanic_Larissa5.csv", header=True)

#PREVISÕES - MODELO XGBOOTS + DUMMIES
dados = ['Sex_binario', 'Age', 'Pclass', 'Fare','Is_alone','SibSp','Parch','Embarked']
x_treino = datatrain[dados].fillna(-1)
y_treino = datatrain['Survived']

x_teste = datatest[dados].fillna(-1)
y_teste = datasuvived['Survived']

treino_x = datatrain[dados].fillna(-1)
teste_x = datatest[dados].fillna(-1)

X_testeDummies = pd.get_dummies(teste_x).astype(int)
Xdummies = pd.get_dummies(treino_x).astype(int)
Ydummies = y_treino

Xdummies = Xdummies.drop('Embarked_-1', axis=1)

X_teste = X_testeDummies.values
X = Xdummies.values
Y = Ydummies.values

data_dmatrix = xgb.DMatrix(data=X,label=Y)
xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X,Y)

preds = xg_reg.predict(X_teste)
pred = [round(value) for value in preds]
accuracy = accuracy_score(y_teste, pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

sub = pd.Series(preds, index=datatest['PassengerId'], name='Survived')
sub.to_csv("Titanic_Larissa4.csv", header=True)