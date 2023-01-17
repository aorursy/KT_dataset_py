import pandas as pd
train = pd.read_csv("../input/train.csv",na_values="?")
train.head()
train = train.drop(["parentesco1","parentesco2","parentesco3","parentesco4","parentesco5","parentesco6",
                    "parentesco7","parentesco8","parentesco9","parentesco10","parentesco11","parentesco12",
                    "estadocivil1","estadocivil2","estadocivil3","estadocivil4","estadocivil5","estadocivil6",
                    "estadocivil7","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin",
                    "SQBovercrowding","SQBdependency","SQBmeaned","agesq","r4h1","r4h2","r4h3","r4m1",
                    "r4m2","r4m3","tamhog","tamviv"],axis=1)
train["n.overcrowding"]=train.hacdor+train.hacapo
train["pared"]=train.paredblolad+2*train.paredzocalo+3*train.paredpreb+4*train.pareddes+5*train.paredmad+6*train.paredzinc+7*train.paredfibras+8*train.paredother
train["piso"]=train.pisomoscer+2*train.pisocemento+3*train.pisoother+4*train.pisonatur+5*train.pisomadera
train["techo"]=train.techozinc+2*train.techoentrepiso+3*train.techocane+4*train.techootro
train["abastagua"]=train.abastaguadentro+2*train.abastaguafuera
train["lugar"]=train.lugar1+2*train.lugar2+3*train.lugar3+4*train.lugar4+5*train.lugar5+6*train.lugar6
train["electricity"]=train.public+2*train.planpri+3*train.coopele
train["sanitario"]=train.sanitario2+2*train.sanitario3+3*train.sanitario5+4*train.sanitario6
train["energcocinar"]=train.energcocinar2+2*train.energcocinar3+3*train.energcocinar4
train["rubbish"]=train.elimbasu1+2*train.elimbasu2+3*train.elimbasu3+4*train.elimbasu4+5*train.elimbasu5+6*train.elimbasu6
train["quality"]=train.epared1+2*train.epared2+3*train.epared3+train.etecho1+2*train.etecho2+3*train.etecho3+train.eviv1+2*train.eviv2+3*train.eviv3
train["education"]=train.instlevel2+2*train.instlevel3+3*train.instlevel4+4*train.instlevel5+5*train.instlevel6+6*train.instlevel7+7*train.instlevel8+8*train.instlevel9
train["appliances"]=train.television+train.computer+train.mobilephone+train.v18q
atrain = train[["n.overcrowding","pared","piso","techo","abastagua","lugar","electricity","sanitario","energcocinar",
                "rubbish","quality","education","appliances","male","rooms","age","Target"]]
train = pd.read_csv("../input/test.csv",na_values="?")
train["n.overcrowding"]=train.hacdor+train.hacapo
train["pared"]=train.paredblolad+2*train.paredzocalo+3*train.paredpreb+4*train.pareddes+5*train.paredmad+6*train.paredzinc+7*train.paredfibras+8*train.paredother
train["piso"]=train.pisomoscer+2*train.pisocemento+3*train.pisoother+4*train.pisonatur+5*train.pisomadera
train["techo"]=train.techozinc+2*train.techoentrepiso+3*train.techocane+4*train.techootro
train["abastagua"]=train.abastaguadentro+2*train.abastaguafuera
train["lugar"]=train.lugar1+2*train.lugar2+3*train.lugar3+4*train.lugar4+5*train.lugar5+6*train.lugar6
train["electricity"]=train.public+2*train.planpri+3*train.coopele
train["sanitario"]=train.sanitario2+2*train.sanitario3+3*train.sanitario5+4*train.sanitario6
train["energcocinar"]=train.energcocinar2+2*train.energcocinar3+3*train.energcocinar4
train["rubbish"]=train.elimbasu1+2*train.elimbasu2+3*train.elimbasu3+4*train.elimbasu4+5*train.elimbasu5+6*train.elimbasu6
train["quality"]=train.epared1+2*train.epared2+3*train.epared3+train.etecho1+2*train.etecho2+3*train.etecho3+train.eviv1+2*train.eviv2+3*train.eviv3
train["education"]=train.instlevel2+2*train.instlevel3+3*train.instlevel4+4*train.instlevel5+5*train.instlevel6+6*train.instlevel7+7*train.instlevel8+8*train.instlevel9
train["appliances"]=train.television+train.computer+train.mobilephone+train.v18q
atest = train[["n.overcrowding","pared","piso","techo","abastagua","lugar","electricity","sanitario","energcocinar",
                "rubbish","quality","education","appliances","male","rooms","age"]]
atrain.head()
atest.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
Xtrain = atrain.drop(["Target"],axis=1)
Ytrain = atrain.Target
values = [1,5,10,15,20,25,30]
scores = []
for k in values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn,Xtrain,Ytrain,cv=10)
    avg = sum(score)/10
    scores.append(avg)
scores
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(Xtrain,Ytrain)
target_pred = knn.predict(atest).tolist()
target_pred[:20]
pd.DataFrame({"Id":train.Id,"Target":target_pred}).to_csv("newprediction.csv",index=False)
