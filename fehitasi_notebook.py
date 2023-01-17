"""
Versão Alpha
"""
import pandas as pd
import sklearn
adult = pd.read_csv("../input/train_data",
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.info()
jj=adult
jj.head()


jj=pd.get_dummies(jj,columns=["sex","race","workclass","occupation","relationship",'marital.status'])
jj.info()
testAdult = pd.read_csv("../input/test_data",
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
j2=testAdult
j2=pd.get_dummies(j2,columns=["sex","race","workclass","occupation","relationship",'marital.status'])
Xadult = jj.dropna()
Xadult = jj[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = jj.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(Xadult,Yadult)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
XtestAdult = j2[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
YtestPred = knn.predict(XtestAdult)

#escreve o codigo em formato txt/csv na mesma pasta do jupyter
ref_arquivo = open("sub.txt","w")
ref_arquivo.write("Id,income\n")

for linha in range(len(YtestPred)):
    ref_arquivo.write("%i"%linha)
    ref_arquivo.write(",")
    ref_arquivo.write(YtestPred[linha])
    ref_arquivo.write("\n")

ref_arquivo.close()

