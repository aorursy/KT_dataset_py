# Imports:
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
# Importando Base Adult - Treino e Teste:
adultTreino = pd.read_csv("../input/traind/train_data.csv",
        engine='python',
        na_values="?")

adultTeste = pd.read_csv("../input/testdat/test_data.csv",
        engine='python',
        na_values="?")
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

adultTreino.shape
# Retirando Missing Data
adultTreino_no_missing_data = adultTreino.dropna()
X = adultTreino_no_missing_data[["age", "relationship", "education.num", "marital.status", "capital.gain", "capital.loss", "race", "sex", "hours.per.week"]]
Ytreino = adultTreino_no_missing_data[["income"]]
Xt = X.apply(preprocessing.LabelEncoder().fit_transform)
scaler = MinMaxScaler()
Xtreino = scaler.fit_transform(Xt)
warnings.filterwarnings('ignore')
score_list = []
maxscore = 0
kmax = 0
for num in np.arange(0.01, 1.2, 0.01).tolist():
    clf = LogisticRegression(C=num)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=5)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        kmax = num
plt.plot(np.arange(0.01, 1.2, 0.01), score_list)
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.show()
print(maxscore, kmax)
score_list = []
maxscore = 0
kmax = 0
for num in range(1, 51):
    clf = RandomForestClassifier(max_depth = num, random_state=1)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=5)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        kmax = num
plt.plot(np.arange(1, 51), score_list)
plt.ylabel('Accuracy')
plt.xlabel('Max Depth')
plt.show()
print(maxscore, kmax)
score_list = []
maxscore = 0
kmax = 0
for num in range(1, 71):
    clf = RandomForestClassifier(n_estimators = num, max_depth = 10, random_state=1)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=5)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        kmax = num
plt.plot(np.arange(1, 71), score_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Estimators')
plt.show()
print(maxscore, kmax)
score_list = []
maxscore = 0
kmax = 0
for num in range(1, 51):
    clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators = 38, max_depth = 10), n_estimators=num)
    scores = cross_val_score(clf, Xtreino, Ytreino, cv=5)
    score_list.append(scores.mean())
    if scores.mean() >= maxscore:
        maxscore = scores.mean()
        kmax = num
plt.plot(np.arange(1, 51), score_list)
plt.ylabel('Accuracy')
plt.xlabel('Number of Estimators')
plt.show()
print(maxscore, kmax)