import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
costa = pd.read_csv(r"../input/train.csv",
        sep=r'\s*,\s*',engine='python',na_values="?")
costa.shape
costa.head()
costa.tail()
ncosta = costa.dropna()
ncosta.shape
numericos = list(costa.select_dtypes(include=[np.number]).columns)
costa.columns[costa.isna().any()].tolist()
colunas = list(set(numericos)- {'Target','v2a1', 'v18q1', 'rez_esc',
                                'meaneduc', 'SQBmeaned'})
xcosta = costa[colunas]
xcosta.head()
xcosta.shape
ycosta = costa.Target
(ycosta.value_counts()/ycosta.size).plot(kind="bar")
ycosta.describe()
teste = pd.read_csv(r"../input/test.csv",
        sep=r'\s*,\s*',engine='python',na_values="?")
teste.head()
xteste = teste[colunas]
xteste.shape
xteste.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
from sklearn.model_selection import cross_val_score
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=10)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=20)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=30)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=40)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=50)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=60)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=70)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=65)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=55)
cross_val_score(knn,xcosta,ycosta,cv=10)
knn = KNeighborsClassifier(n_neighbors=60)
knn.fit(xcosta,ycosta)
ypred = knn.predict(xteste)
ypred
Pred=pd.DataFrame(columns = ['Id','Target'])
Pred['Target'] = ypred
Pred['Id'] = teste.Id
Pred.to_csv("Costa_Ypred.csv",index=False)
Pred.shape
Pred.tail()
(Pred.Target.value_counts()/Pred.size).plot(kind="bar")