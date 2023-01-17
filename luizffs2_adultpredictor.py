'''PMR3508 - Aprendizado de máquina e reconhecimento de padrões
   Análise da base Adult
   Autor: Luiz Fernando Ferreira da Silva
'''
import pandas as pd
import sklearn
base = pd.read_csv("../input/bases-de-dados/train_data.csv",na_values="?")

base.head()
import matplotlib.pyplot as graphic
base["income"].value_counts().plot(kind="bar")
base["relationship"].value_counts().plot(kind="pie")
base["occupation"].value_counts().plot(kind="bar")
base["race"].value_counts().plot(kind="pie")
#Analizando relações que dados numéricos e não numéricos tem com a condição 'income'


def percent(colum):
    return colum*100//float(colum[-1])
targetxsex = pd.crosstab(base["income"], base["sex"],margins = True)
targetxsex

targetxsex.apply(percent,axis=0)
no_number = ["Male","Female"]
number = [3,1]
targetxsex.apply(percent,axis=0).plot(kind="bar")
targetxrace = pd.crosstab(base["income"],base["race"],margins=True)
targetxrace.apply(percent,axis=0)
races=["Asian-Pac-Islander","White","Black","Amer-Indian-Eskimo","Other"]
perc=["26", "25", "12", "11","9"]
no_number += races
number += perc
targetxmaritalstatus = pd.crosstab(base["income"],base["marital.status"],margins=True)
targetxmaritalstatus.apply(percent,axis=0)
status=["Married-AF-spouse", "Married-civ-spouse", "Divorced", "Married-spouse-absent", "Widowed", 
        "Separated", "Never-married"]
perc=["44", "43", "10", "8", "8", "6", "4"]
no_number += status
number += perc
targetxrelationship = pd.crosstab(base["income"],base["relationship"],margins=True)
targetxrelationship.apply(percent,axis=0)
relations=["Wife", "Husband", "Not-in-family", "Unmarried", "Other-relative", "Own-child"]
perc=["47", "44", "10", "6", "3", "1"]
no_number += relations
number += perc
targetxeducationnum = pd.crosstab(base["income"],base["education.num"],margins=True)
targetxeducationnum.apply(percent,axis=0)
targetxeducationnum = pd.crosstab(base["education.num"],base["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducation = pd.crosstab(base["income"],base["education"],margins=True)
targetxeducation.apply(percent,axis=0)
targetxage = pd.crosstab(base["age"],base["income"],margins=True)
targetxage.apply(percent,axis=1).plot()
targetxoccupation = pd.crosstab(base["income"],base["occupation"],margins=True)
targetxoccupation.apply(percent,axis=0)
occupation = ["Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners",
              "Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales",
              "Tech-support","Transport-moving"]
perc = ["13","11","22","48","11","6","12","4","0","44","32","26","30","20"]
no_number+=occupation
number+=perc
targetxworkclass = pd.crosstab(base["income"],base["workclass"],margins=True)
targetxworkclass.apply(percent,axis=0)
work=["Self-emp-inc", "Federal-gov", "Local-gov", "Self-emp-not-inc", "State-gov", "Private", "Without-pay", "Never-worked"]
perc=["55", "38", "29", "28", "27", "21", "0", "0"]
no_number += work
number += perc
"""Esta função substitui um dado não numérico por um número relacionado com a porcentagem de pessoas que apresentam aquele dado
   e a característica ">50k"
"""
def num_func(label):
    for i in range(len(number)):
        if label == no_number[i]:
            return number[i]
    return label
base["sex"] = base["sex"].apply(num_func)
base["workclass"] = base["workclass"].apply(num_func)
base["marital.status"] = base["marital.status"].apply(num_func)
base["relationship"] = base["relationship"].apply(num_func)
base["occupation"] = base["occupation"].apply(num_func)
base["race"] = base["race"].apply(num_func)
nbase = base.dropna()
nbase
Xnbase = nbase[["age","workclass","education.num","marital.status","occupation","relationship","race",
                "sex","capital.gain","capital.loss","hours.per.week"]]
Xnbase
Ynbase = nbase.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=23)  #Depois de vários testes decidi deixar n_neighbors=23
from sklearn.model_selection import cross_val_score as cross
scores = cross(knn, Xnbase, Ynbase, cv=10)
scores
arquivo2 = '../input/bases-de-dados/test_data.csv'
tester = pd.read_csv(arquivo2,na_values="?")
tester["sex"] = tester["sex"].apply(num_func)
tester["workclass"] = tester["workclass"].apply(num_func)
tester["marital.status"] = tester["marital.status"].apply(num_func)
tester["relationship"] = tester["relationship"].apply(num_func)
tester["occupation"] = tester["occupation"].apply(num_func)
tester["race"] = tester["race"].apply(num_func)
ntester = tester.dropna()
Xntester = ntester[["age","workclass","education.num","marital.status","occupation","relationship","race",
                "sex","capital.gain","capital.loss","hours.per.week"]]
knn.fit(Xnbase,Ynbase)
Ytestpred = knn.predict(Xntester)
Ytestpred
from sklearn.metrics import accuracy_score as acs
acs(Ynbase,knn.predict(Xnbase))