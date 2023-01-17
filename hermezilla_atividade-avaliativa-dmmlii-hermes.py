# Importando todas as bibliotecas

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
#Carregando o banco
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
df.head(20)
#verificando se o tipo dos dados no banco está em conformidade com o dicionário
df.info()
#verificando a quantidade de valores faltantes dos campos
df.isnull().sum()
#Boxplots dos valores das dívidas contraídas em função da razão da hipoteca. Estão comparados pelo pagamento ou não.
sns.boxplot(x="REASON", y="LOAN",
            hue="BAD", palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)
#Scatterplot do valor da propriedade com o valor atual da hipoteca
plt.plot(range(500000))
ax = sns.scatterplot(x="VALUE", y="MORTDUE",
                     hue="BAD",
                     data=df)
plt.xlim(0, 500000)
plt.ylim(0, 500000)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()
#Embonecando as variáveis categóricas
df = pd.get_dummies(df, columns=['REASON','JOB'])
df.head().T
#Definindo as variáveis independentes
feats = [c for c in df.columns if c not in ['BAD']]
#Separando a base em treino e teste
train, valid = train_test_split(df, test_size=0.2, random_state=42)

train.shape, valid.shape
#Rodando o primeiro modelo
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(train[feats], train['BAD'])
#Imputação
df.fillna(-1, inplace=True)
df.isnull().sum()
#Refazendo a separação
train, valid = train_test_split(df, test_size=0.2, random_state=42)

train.shape, valid.shape
#Rodando o primeiro modelo novamente
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(train[feats], train['BAD'])
#Aplicando o modelo na base de validação e verificando a acurácia
preds_val = rf.predict(valid[feats])

accuracy_score(valid['BAD'], preds_val)
#Mostrando a matriz de confusão
skplt.metrics.plot_confusion_matrix(valid['BAD'],preds_val)
#Testando o limitador de profundidade da árvore
for i in range(1,11,1):
    rft = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=i)
    rft.fit(train[feats], train['BAD'])
    pred_teste = rft.predict(valid[feats])
    print(str(i)+" de profundidade: "+str(accuracy_score(valid['BAD'], pred_teste)))
#Testando o número de estimadores
for i in range(1000,100,-100):
    rft = RandomForestClassifier(n_estimators=i, random_state=42)
    rft.fit(train[feats], train['BAD'])
    pred_teste = rft.predict(valid[feats])
    print(str(i)+": "+str(accuracy_score(valid['BAD'], pred_teste)))
#Verificando o desbalanceio da variável dependente
df['BAD'].value_counts()
#Testando colocar pesos nas possibilidades pagadores para atacar o desbalanceio
class_weight = dict({1:4, 0:1})
rdf = RandomForestClassifier(bootstrap=True,
            class_weight=class_weight, 
            criterion='gini',
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=200,
            oob_score=False,
            random_state=42,
            verbose=0, warm_start=False)

rdf.fit(train[feats], train['BAD'])

pred_teste = rdf.predict(valid[feats])
print(accuracy_score(valid['BAD'], pred_teste))
# cria o vetor de notas, mostra e mostra a média
scores = cross_val_score(rf, df[feats], df['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
# cria um objeto xgb
xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, learning_rate=0.05)
#Usa o cross validation como antes, mas com o xgb
scores = cross_val_score(xgb, df[feats], df['BAD'], n_jobs=-1, cv=5)

scores, scores.mean()
#Cria um dicionário com os tipos de parâmetros que serão testados
grid_param = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}
#Cria o objeto grid search utilizando o dicionário anterior
gd_sr = GridSearchCV(estimator=rf,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)
#Treina o modelo testando combinações de todos os parâmetros (demorado)
gd_sr.fit(df[feats], df['BAD'])
#Mostra os melhores parâmetros
best_parameters = gd_sr.best_params_
print(best_parameters)
#Mostra a acurácia com os melhores parâmetros
best_result = gd_sr.best_score_
print(best_result)