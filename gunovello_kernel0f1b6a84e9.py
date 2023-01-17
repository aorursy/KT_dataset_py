import pandas as pd
import sklearn
import matplotlib.pyplot as plt
adult = pd.read_csv('../input/badult/train_data.csv',
                  sep=',', engine='python',
                  na_values="?")
adult.shape
adult.head()
adult.tail()
nadult = adult.dropna()
nadult.shape
Target = nadult[nadult['income'] == '>50K']
nadult["education"].value_counts().plot(kind="bar")
Target["education"].value_counts().plot(kind="bar")
nadult["native.country"].value_counts()
Target["native.country"].value_counts()
(nadult["native.country"].value_counts()-Target["native.country"].value_counts())/nadult["native.country"].value_counts()
(Target["native.country"].value_counts()/nadult["native.country"].value_counts())
(Target["native.country"].value_counts()/nadult["native.country"].value_counts()).plot(kind="bar")
nadult.describe()
xadult = adult[["age","education.num","capital.gain",
                "capital.loss","hours.per.week"]]
yadult = adult.income
teste = pd.read_csv(r"../input/badult/test_data.csv",
        sep=r'\s*,\s*',engine='python',na_values="?")
teste.head()
nteste = teste.dropna()
xteste = teste[["age","education.num","capital.gain",
                "capital.loss","hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
from sklearn.model_selection import cross_val_score
cross_val_score(knn,xadult,yadult,cv=10)
knn = KNeighborsClassifier(n_neighbors=10)
cross_val_score(knn,xadult,yadult,cv=10)
knn = KNeighborsClassifier(n_neighbors=20)
cross_val_score(knn,xadult,yadult,cv=10)
knn = KNeighborsClassifier(n_neighbors=30)
cross_val_score(knn,xadult,yadult,cv=10)
knn = KNeighborsClassifier(n_neighbors=40)
cross_val_score(knn,xadult,yadult,cv=10)
knn = KNeighborsClassifier(n_neighbors=32)
cross_val_score(knn,xadult,yadult,cv=10)
knn.fit(xadult,yadult)
ypred = knn.predict(xteste)
(yadult.value_counts()/yadult.size).plot(kind="bar")
n=0
i=0
Y=[]
for elemento in ypred:
    if elemento == '<=50K':
        n=n+1
        Y.append([i,'<=50K'])
    else:
        Y.append([i,'>50K'])
    i=i+1
n/len(ypred)
Pred=pd.DataFrame(Y,columns = ['Id','income'])
Pred.to_csv("Ypred.csv",index=False)
Pred
teste.tail()
