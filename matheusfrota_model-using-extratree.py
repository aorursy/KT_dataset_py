#Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# Open data set

df_train = pd.read_csv('../input/train.csv')
df_test =  pd.read_csv('../input/test.csv')

df_train.head()
df_test.head()
df_train.info()
df_train.describe()
df_train.duplicated().sum()
# Para suavizar a presença e o efeito dos outliers nos modelos, irei aplicar uma transformação nos dados numéricos.

# Padronizando os dados com StandartScaler
from sklearn.preprocessing import StandardScaler


# Utilizando apenas as colunas numéricas que não foram codificadas como a do tipo de solo e a designação da área.
colunas = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
           'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
           'Horizontal_Distance_To_Fire_Points','Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 
           'Wilderness_Area4','Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 
           'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
           'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 
           'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 
           'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 
            'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']


scaler = StandardScaler()
#Treino

scaler.fit(df_train[colunas])
df_train[colunas] = scaler.transform(df_train[colunas])

#Teste

scaler.fit(df_test[colunas])
df_test[colunas] = scaler.transform(df_test[colunas])
df_train.head()
# Correlação das colunas numéricas

df_train[colunas].corr()
#Removendo atributos

df_train = df_train.drop(['Soil_Type7','Soil_Type15'],1)

df_test = df_test.drop(['Soil_Type7','Soil_Type15'],1)
# Vizualização dos dados

# Depois de cuidar da assimetria dos dados, vamos começar as vizualisações para tentar entender melhor o problema

#Gráfico 1
df_train.plot(x = 'Aspect', y = 'Hillshade_3pm',c = ('red','blue'),kind = 'scatter', figsize = (5,5));

#Gráfico 2
df_train.plot(x = 'Slope', y = 'Hillshade_Noon',c = ('red','blue'),kind = 'scatter', figsize = (5,5));
#Gráfico 3
df_train.plot(x = 'Horizontal_Distance_To_Hydrology', y = 'Vertical_Distance_To_Hydrology',c = ('red','blue'),kind = 'scatter', figsize = (5,5));

#Gráfico 4
df_train.plot(x = 'Hillshade_9am', y = 'Hillshade_3pm',c = ('red','blue'),kind = 'scatter', figsize = (5,5));

#Gráfico 5
df_train.plot(x = 'Hillshade_Noon', y = 'Hillshade_3pm',c = ('red','blue'),kind = 'scatter', figsize = (5,5));
# Importando as bibliotecas necessárias

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# Distribuição do atributo Cover-Type

print(Counter(df_train['Cover_Type']))
# Preparação das variáveis

X = df_train.drop(['Cover_Type','Id'],1)
Y = df_train['Cover_Type']

# Métricas

seed = 42
scoring = 'accuracy'
validation_size = 0.30
# Separando em Teste e Treino

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state = seed)
# Vamos verificar o comportamento desses 3 algoritmos

models = []
models.append(('RF',RandomForestClassifier()))
models.append(('ETC',ExtraTreesClassifier()))
models.append(('KNN', KNeighborsClassifier()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Comparando os algorítmos

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Grid Search
#model = ExtraTreesClassifier(random_state = 0) 

# Grid search parâmetros
#param_grid = {
 #   "n_estimators": [500, 550, 600, 650, 700, 750, 800 , 850, 900, 950]
#}

# Executando grid search
#CV_model = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=KFold(n_splits=10, random_state=seed))
#CV_model_result = CV_model.fit(X_train, Y_train)

# Resultados
#print("Best: %f using %s" % (CV_model_result.best_score_, CV_model_result.best_params_))
#Avaliando a precisão do modelo ETC depois de ter encontrado o melhor parâmetro

ETC = ExtraTreesClassifier( n_estimators = 550)
ETC.fit(X_train, Y_train)
predicoes1 = ETC.predict(X_test)
print("Precisão: {} \n".format(accuracy_score(Y_test, predicoes1)))
print(confusion_matrix(Y_test, predicoes1))
print(classification_report(Y_test, predicoes1))
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
predicoes2 = RF.predict(X_test)
print("Precisão: {} \n".format(accuracy_score(Y_test, predicoes2)))
print(confusion_matrix(Y_test, predicoes2))
print(classification_report(Y_test, predicoes2))
# Banco de dados de teste
X_tes = df_test.drop(['Id'],1)

#Previsões com os valores do banco de dados teste
test_predicoes1 = ETC.predict(X_tes)
test_predicoes2 = RF.predict(X_tes)
valores_pred = np.column_stack((predicoes1, predicoes2))
valores_pred_test = np.column_stack((test_predicoes1, test_predicoes2))
# Modelo final
modelo_fin = ExtraTreesClassifier( n_estimators = 550)
modelo_fin.fit(valores_pred,Y_test)
#Predição final
predicao_final = modelo_fin.predict(valores_pred_test)
# Acuracia do modelo final
modelo_fin.score(valores_pred, Y_test)
my_submission = pd.DataFrame({'Id': df_test.Id, 'Cover_Type': predicao_final})
my_submission.to_csv('submission11.csv', index=False)