# Imports:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import warnings
# Importando Base Adult - Treino e Teste:
adultTreino = pd.read_csv("../input/train-data/train_data.csv",
        engine='python',
        na_values="?")

adultTeste = pd.read_csv("../input/test-data/test_data.csv",
        engine='python',
        na_values="?")
adultTreino
adultTreino["sex"].value_counts().plot(kind="bar")
adultTreino["race"].value_counts().plot(kind="bar")
# Extraindo as observações em que o indivíduo ganha mais que 50K anuais:
more50K = adultTreino[(adultTreino['income'] == ">50K")]

# Extraindo as observações em que o indivíduo ganha menos que 50K anuais:
less50K = adultTreino[(adultTreino['income'] == "<=50K")]
# Influência do sexo nos indivíduos que ganham mais que 50K:
more50K["sex"].value_counts().plot(kind="bar")
more50K["sex"].value_counts()
# Influência do sexo nos indivíduos que ganham menos que 50K:
less50K["sex"].value_counts().plot(kind="bar")
less50K["sex"].value_counts()
# Influência da educação nos indivíduos que ganham mais que 50K:
more50K["education"].value_counts().plot(kind="bar")
more50K["education"].value_counts()
# Influência da educação nos indivíduos que ganham menos que 50K:
less50K["education"].value_counts().plot(kind="bar")
less50K["education"].value_counts()
# Influência da raça nos indivíduos que ganham mais que 50K:
more50K["race"].value_counts().plot(kind="bar")
more50K["race"].value_counts()
# Influência da raça nos indivíduos que ganham menos que 50K:
less50K["race"].value_counts().plot(kind="bar")
less50K["race"].value_counts()
# Influência do relacionamento nos indivíduos que ganham mais que 50K:
more50K["relationship"].value_counts().plot(kind="bar")
more50K["relationship"].value_counts()
# Influência do relacionamento nos indivíduos que ganham menos que 50K:
less50K["relationship"].value_counts().plot(kind="bar")
less50K["relationship"].value_counts()
# Influência da classe de trabalho nos indivíduos que ganham mais que 50K:
more50K["workclass"].value_counts().plot(kind="bar")
more50K["workclass"].value_counts()
# Influência da classe de trabalho nos indivíduos que ganham menos que 50K:
less50K["workclass"].value_counts().plot(kind="bar")
less50K["workclass"].value_counts()
# Influência da ocupação nos indivíduos que ganham mais que 50K:
more50K["occupation"].value_counts().plot(kind="bar")
more50K["occupation"].value_counts()
# Influência da ocupação nos indivíduos que ganham menos que 50K:
less50K["occupation"].value_counts().plot(kind="bar")
less50K["occupation"].value_counts()
# Retirando Missing Data
adultTreino_no_missing_data = adultTreino.dropna()
adultTreino_no_missing_data.shape
# Classificação com algumas features aparentemente interessantes:
X = adultTreino_no_missing_data[["age", "relationship", "education.num", "marital.status", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Ytreino = adultTreino_no_missing_data[["income"]]
Xtreino = X.apply(preprocessing.LabelEncoder().fit_transform)
scaler = MinMaxScaler()
XtreinoScaler = scaler.fit_transform(Xtreino)
# Teste com scaler e distância de Mahalanobis:
warnings.filterwarnings('ignore')
cov = np.cov(XtreinoScaler, rowvar=False)
clf = KNeighborsClassifier(n_neighbors=19, metric='mahalanobis', metric_params={'V':cov})
scores = cross_val_score(clf, XtreinoScaler, Ytreino, cv=10)
scores.mean()
# Verificação da influência do k na classificação pelo algoritmo de k-nearest neighbor:
warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
nmax = 0
for n in range (1, 50):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, XtreinoScaler, Ytreino, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        nmax = n
plt.plot(np.arange(1, 50),score_list)
plt.show()
print(maxscore, nmax) 
# Classificação com algumas features aparentemente interessantes com inclusão de Estado Civil:
X = adultTreino_no_missing_data[["age", "education.num", "relationship", "marital.status", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Ytreino = adultTreino_no_missing_data[["income"]]
Xtreino = X.apply(preprocessing.LabelEncoder().fit_transform)
# Verificação da influência do k na classificação pelo algoritmo de k-nearest neighbor:
warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
nmax = 0
for n in range (1, 50):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        nmax = n
plt.plot(np.arange(1, 50),score_list)
plt.show()
print(maxscore, nmax) 
# Classificação com algumas features aparentemente interessantes com inclusão da classe de trabalho:
X = adultTreino_no_missing_data[["age", "education.num", "relationship", "workclass", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Ytreino = adultTreino_no_missing_data[["income"]]
Xtreino = X.apply(preprocessing.LabelEncoder().fit_transform)
# Verificação da influência do k na classificação pelo algoritmo de k-nearest neighbor:
warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
nmax = 0
for n in range (1, 50):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        nmax = n
plt.plot(np.arange(1, 50),score_list)
plt.show()
print(maxscore, nmax) 
# Classificação com outras features aparentemente interessantes:
X = adultTreino_no_missing_data[["age", "education", "relationship", "occupation", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Ytreino = adultTreino_no_missing_data[["income"]]
Xtreino = X.apply(preprocessing.LabelEncoder().fit_transform)
# Verificação da influência do k na classificação pelo algoritmo de k-nearest neighbor:
warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
nmax = 0
for n in range (1, 50):
    clf = KNeighborsClassifier(n_neighbors=n)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=10)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        nmax = n
plt.plot(np.arange(1, 50),score_list)
plt.show()
print(maxscore, nmax) 
adultTeste.shape
# retirando Missing Data da base de teste:
adultTeste_no_missing_data = adultTeste.dropna()
adultTeste_no_missing_data.shape
adultTeste_no_missing_data
X = adultTreino_no_missing_data[["age", "relationship", "education.num", "marital.status", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Ytreino = adultTreino_no_missing_data[["income"]]
Xtreino1 = X.apply(preprocessing.LabelEncoder().fit_transform)
Xtreino = scaler.fit_transform(Xtreino1)
# "Fitando" o classificador:
clf = KNeighborsClassifier(n_neighbors=19)
clf.fit(Xtreino, Ytreino)
# Preparando features de teste para classificação:
Xtest = adultTeste_no_missing_data[["age", "relationship", "education.num", "marital.status", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Xteste1 = Xtest.apply(preprocessing.LabelEncoder().fit_transform)
Xteste = scaler.fit_transform(Xteste1)
# Classificando base de teste:
YtestePred = clf.predict(Xteste)
#np.set_printoptions(threshold=np.nan)
#YtestePred
YtestePred.shape
# Preparando features de teste para classificação (observações com Missing Data):
Xtest2 = adultTeste[["age", "relationship", "education.num", "marital.status", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Xteste2_1 = Xtest2.apply(preprocessing.LabelEncoder().fit_transform)
Xteste2 = scaler.fit_transform(Xteste2_1)
# Classificando base de teste com Missing data:
YtestePred2 = clf.predict(Xteste2)
YtestePred2.shape
conv_arr= adultTeste.values
arr1 = np.delete(conv_arr,[1,2,3,4,5,6,7,8,9,10,11,12,13,14],axis=1)
arr1 = arr1.ravel()
dataset = pd.DataFrame({'Id':arr1[:],'Income':YtestePred2[:]})
print(dataset)
dataset.to_csv("submission.csv", index = False)