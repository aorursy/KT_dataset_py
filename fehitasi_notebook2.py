
import pandas as pd
import sklearn
import matplotlib.pyplot as graphic
from sklearn.neighbors import KNeighborsClassifier
import os
print(os.listdir("../input/"))
adult = pd.read_csv("../input/teste-ola/train_data", 
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.info()

jj=adult
jj.head()




jj["marital.status"].value_counts().plot(kind="pie")


jj["race"].value_counts().plot(kind="pie")
jj["sex"].value_counts().plot(kind="pie")
jj["relationship"].value_counts().plot(kind="pie")
jj["education"].value_counts().plot(kind="pie")
jj["occupation"].value_counts().plot(kind="pie")
jj["workclass"].value_counts().plot(kind="pie")
jj=pd.get_dummies(jj,columns=["sex","race","workclass","relationship",'marital.status',"occupation"])
jj.info()
testAdult = pd.read_csv("../input/teste-ola/train_data",
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

j2=testAdult
testAdult
j2
j2=pd.get_dummies(j2,columns=["sex","race","workclass","relationship",'marital.status',"occupation"])
Xadult = jj.dropna()
Xadult = jj[["age","education.num","capital.gain","capital.loss" ,"hours.per.week","sex_Female", "relationship_Husband", "race_White","workclass_Private",
            "occupation_Adm-clerical","relationship_Wife", "occupation_Prof-specialty", "occupation_Craft-repair","marital.status_Married-civ-spouse",
            "relationship_Own-child","marital.status_Divorced"]]
Yadult = jj.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=18, p=1)

knn.fit(Xadult,Yadult)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
XtestAdult = j2[["age","education.num","capital.gain","capital.loss" ,"hours.per.week","sex_Female", "relationship_Husband", "race_White",
                "workclass_Private","occupation_Adm-clerical","relationship_Wife", "occupation_Prof-specialty", "occupation_Craft-repair",
                "marital.status_Married-civ-spouse","relationship_Own-child","marital.status_Divorced"]]
#XtestAdult=XtestAdult.fillna(0)

YtestPred = knn.predict(XtestAdult)

ref_arquivo = open("sub.txt","w")
ref_arquivo.write("Id,income\n")

for linha in range(len(YtestPred)):
    ref_arquivo.write("%i"%linha)
    ref_arquivo.write(",")
    ref_arquivo.write(YtestPred[linha])
    ref_arquivo.write("\n")

ref_arquivo.close()


