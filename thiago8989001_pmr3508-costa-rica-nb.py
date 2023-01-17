# Imports:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
# Importando Base HouseholdIncome - Treino e Teste:
treino = pd.read_csv("../input/treino-data/train.csv",
        engine='python',
        na_values="NaN")

teste = pd.read_csv("../input/teste-data/test.csv",
        engine='python',
        na_values="NaN")
treino
# Verificando quantidade de observações pertencentes a cada classe do dataset:
treino.Target.value_counts().plot(kind="bar")
treino.Target.value_counts()
# Contando quantidade de dados faltantes relativa a cada feature:
treino.isnull().sum()
# Verifica indice das features com mais de 1000 dados faltantes:
i = 0
nan = []
indice = []
for i in range(0, 143):
    column = treino.iloc[:,i]
    nan.append(column.isnull().sum())
    if column.isnull().sum()>1000:
        indice.append(i)
indice
# Nova base de treino sem as features com alta quantidade de dados faltantes:
newTreino = treino.drop(['v2a1','v18q1','rez_esc'], axis=1)
newTreino
#Contando quantidade de dados faltantes relativa a cada feature da nova base:
newTreino.isnull().sum()
# Elimina observações com dados faltantes da base de treino obtida anteriormente:
treino_no_missing = newTreino.dropna()
treino_no_missing
# Seleciona as features da base e aplica pre processamento para transformar features não numéricas em numéricas
X = treino_no_missing.iloc[:,1:138]
Y = treino_no_missing.Target
X2 = X.apply(preprocessing.LabelEncoder().fit_transform)
# Create and fit selector
#selector = SelectKBest(f_classif, k=5)
selector = SelectKBest(chi2, k=5)
X_new = selector.fit_transform(X2, Y)
# Get idxs of columns to keep
idxs_selected = selector.get_support(indices=True)
# Create new dataframe with only desired columns, or overwrite existing
new_features = X2.columns[idxs_selected]
dataframe = pd.DataFrame(X_new, columns=new_features)
dataframe
# Usando a função SelectKBest com teste ANOVA, seleciona-se um certo número de features que de melhor acurácia:
warnings.filterwarnings('ignore')
clf = KNeighborsClassifier(n_neighbors=71)
features = 0
score_list = []
maxscore = 0
max_features = 0
for features in range(1, 138):
    selector = SelectKBest(f_classif, k=features)
    X_aux = selector.fit_transform(X2, Y)
    idxs_selected = selector.get_support(indices=True)
    new_features = X2.columns[idxs_selected]
    X_new = pd.DataFrame(X_aux, columns=new_features)
    scores = cross_val_score(clf, X_new, Y, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        max_features = features
plt.plot(np.arange(1, 138),score_list)
plt.show()
print(maxscore, max_features) 
# Verifica-se influência do k do algoritmo de kNN para que de a melhor acurácia:
selector = SelectKBest(f_classif, k=27)
X_aux = selector.fit_transform(X2, Y)
idxs_selected = selector.get_support(indices=True)
new_features = X2.columns[idxs_selected]
X_new = pd.DataFrame(X_aux, columns=new_features)

warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
nmax = 0
for n in range (1, 100):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, X_new, Y, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        nmax = n
plt.plot(np.arange(1, 100),score_list)
plt.show()
print(maxscore, nmax)
# Realiza os mesmos testes apresentados anteriormente com a base completa (sem retirar dados faltantes):
X_completo = treino.iloc[:,1:141]
Y_completo = treino.Target
X_completo_num = X_completo.apply(preprocessing.LabelEncoder().fit_transform)
clf = KNeighborsClassifier(n_neighbors=68)
features = 0
score_list = []
maxscore = 0
max_features = 0
for features in range(1, 141):
    selector = SelectKBest(f_classif, k=features)
    X_aux = selector.fit_transform(X_completo_num, Y_completo)
    idxs_selected = selector.get_support(indices=True)
    new_features = X_completo_num.columns[idxs_selected]
    X_new = pd.DataFrame(X_aux, columns=new_features)
    scores = cross_val_score(clf, X_new, Y_completo, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        max_features = features
plt.plot(np.arange(1, 141),score_list)
plt.show()
print(maxscore, max_features) 
selector = SelectKBest(f_classif, k=29)
X_aux = selector.fit_transform(X_completo_num, Y_completo)
idxs_selected = selector.get_support(indices=True)
new_features = X_completo_num.columns[idxs_selected]
X_new = pd.DataFrame(X_aux, columns=new_features)

warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
nmax = 0
for n in range (1, 100):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, X_new, Y_completo, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        nmax = n
plt.plot(np.arange(1, 100),score_list)
plt.show()
print(maxscore, nmax)
selector = SelectKBest(f_classif, k=29)
X_aux = selector.fit_transform(X_completo_num, Y_completo)
idxs_selected = selector.get_support(indices=True)
new_features = X_completo_num.columns[idxs_selected]
X_new = pd.DataFrame(X_aux, columns=new_features)

# "Fitando" o classificador:
clf = KNeighborsClassifier(n_neighbors=68)
clf.fit(X_new, Y_completo)
new_features
X_test = teste[['hacdor', 'rooms', 'v18q', 'r4h1', 'r4m1', 'r4t1', 'escolari','paredblolad', 'pisomoscer', 'pisocemento', 'cielorazo', 'epared1','epared3', 'etecho1', 'etecho3', 'eviv1', 'eviv3', 'hogar_nin','meaneduc', 'instlevel8', 'overcrowding', 'computer', 'qmobilephone','SQBescolari', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding','SQBdependency', 'SQBmeaned']]
X_teste = X_test.apply(preprocessing.LabelEncoder().fit_transform)
# Classificando base de teste:
YtestePred = clf.predict(X_teste)
arr1= teste.iloc[:,0].values
arr1 = arr1.ravel()
dataset = pd.DataFrame({'Id':arr1[:],'Income':YtestePred[:]})
print(dataset)
dataset.to_csv("submition.csv", index = False)
dataset["Income"].value_counts().plot(kind="bar")
dataset["Income"].value_counts()
