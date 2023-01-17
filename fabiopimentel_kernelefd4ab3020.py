import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
df = pd.read_csv("../input/train-data/train.csv", sep=r'\s*,\s*', engine='python', na_values="?")
df.columns
df.isna().sum()
dfnew = df.drop(columns = ['v2a1','v18q1','rez_esc'])
dfnew.isna().sum()
dfnew.dropna()
dfnew.shape
dfsemid = dfnew.drop(columns = ['Id','Target'])
dfsemid = dfsemid.apply(preprocessing.LabelEncoder().fit_transform)
Ydf = dfnew.Target
Xdf = dfsemid
knn = KNeighborsClassifier(n_neighbors=140)
# gera o classificador
scores = cross_val_score(knn, Xdf, Ydf, cv=10)
# faz a validação cruzada do modelo afim de descobrir a acuracia do modelo antes de testa-lo na base de teste
scores.mean()
knn.fit(Xdf,Ydf)
dft = pd.read_csv("../input/train-data/test.csv", sep=r'\s*,\s*', engine='python', na_values="?")
dft= dft.drop(columns = ['v2a1','v18q1','rez_esc'])
dfsemid = dft.drop(columns = ['Id'])
dfx = dfsemid.apply(preprocessing.LabelEncoder().fit_transform)
YtestPred = knn.predict(dfx)
FinalTestAdult = dfx.assign(Target = YtestPred)
FinalTestAdult = FinalTestAdult.join(dft, lsuffix='_caller', rsuffix='_other')
FinalTestAdult = FinalTestAdult[["Id", "Target"]]
FinalTestAdult.to_csv("sample_submission.csv", index = False, header = True)