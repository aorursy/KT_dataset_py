
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as graphic
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
print(os.listdir("../input/"))
adult = pd.read_csv("../input/train_data.csv", 
        
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
import seaborn as sns
corr = jj[["Id", "age","fnlwgt","education.num","capital.gain","capital.loss",
              "hours.per.week"]]

mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f, ax = plt.subplots(figsize=(17,17))
cmap=sns.diverging_palette(220, 10, as_cmap=False)
sns.heatmap(corr, mask=mask, cmap =cmap, vmax=1, vmin=-1, center=0, square=True, linewidth=.5,cbar_kws={"shrink":.5})
corr = jj.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True), square=True, ax=ax)
sns.pairplot(jj, vars=["Id", "age","fnlwgt","education.num","capital.gain","capital.loss",
              "hours.per.week"])
jj=pd.get_dummies(jj,columns=["sex","race","workclass","relationship",'marital.status',"occupation"])
jj.info()
testAdult = pd.read_csv("../input/test_data.csv",
        
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

ref_arquivo = open("subKNN.csv","w")
ref_arquivo.write("Id,income\n")

for linha in range(len(YtestPred)):
    ref_arquivo.write("%i"%linha)
    ref_arquivo.write(",")
    ref_arquivo.write(YtestPred[linha])
    ref_arquivo.write("\n")

ref_arquivo.close()
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xadult, Yadult)
dtct=clf.predict(XtestAdult)
ref_arquivo = open("subDT.csv","w")
ref_arquivo.write("Id,income\n")

for linha in range(len(dtct)):
    ref_arquivo.write("%i"%linha)
    ref_arquivo.write(",")
    ref_arquivo.write(dtct[linha])
    ref_arquivo.write("\n")

ref_arquivo.close()
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(Xadult, Yadult)
svmc=clf.predict(XtestAdult)
ref_arquivo = open("subSVM.csv","w")
ref_arquivo.write("Id,income\n")

for linha in range(len(svmc)):
    ref_arquivo.write("%i"%linha)
    ref_arquivo.write(",")
    ref_arquivo.write(svmc[linha])
    ref_arquivo.write("\n")

ref_arquivo.close()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15, 2), random_state=1)
clf.fit(Xadult, Yadult) 
nn=clf.predict(XtestAdult)
ref_arquivo = open("subNN.csv","w")
ref_arquivo.write("Id,income\n")

for linha in range(len(nn)):
    ref_arquivo.write("%i"%linha)
    ref_arquivo.write(",")
    ref_arquivo.write(nn[linha])
    ref_arquivo.write("\n")

ref_arquivo.close()