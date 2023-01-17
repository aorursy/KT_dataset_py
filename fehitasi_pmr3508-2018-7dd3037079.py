import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
entrada=[]
entrada2="age","rooms","r4h1","r4h2","tamhog","computer","television"
costa= pd.read_csv("../input/train.csv",
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
costa.info(max_cols=143)
costa.head()
costa.describe()
testcosta = pd.read_csv("../input/test.csv",
        
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train1=costa.dropna()
Xtrain2=train1[["v2a1","lugar1","lugar2","lugar3","lugar4","lugar5","rooms","v18q","r4t1","r4t3","tamhog","escolari","cielorazo","abastaguadentro","abastaguafuera","abastaguano","public","sanitario2",
                "sanitario3","energcocinar2","energcocinar3","elimbasu1","elimbasu3","epared1","epared2","etecho1","etecho2","eviv1","eviv2","male","parentesco1",
                "parentesco3","hogar_nin","hogar_adul","hogar_mayor","hogar_total","meaneduc","instlevel5","instlevel7","instlevel8","instlevel9",
                "bedrooms","overcrowding","tipovivi1","tipovivi2","tipovivi3","tipovivi4"]]
Xtest2=testcosta[["v2a1","lugar1","lugar2","lugar3","lugar4","lugar5","rooms","v18q","r4t1","r4t3","tamhog","escolari","cielorazo","abastaguadentro","abastaguafuera","abastaguano","public","sanitario2",
                  "sanitario3","energcocinar2","energcocinar3","elimbasu1","elimbasu3","epared1","epared2","etecho1","etecho2","eviv1","eviv2","male","parentesco1",
                  "parentesco3","hogar_nin","hogar_adul","hogar_mayor","hogar_total","meaneduc","instlevel5","instlevel7","instlevel8","instlevel9",
                  "bedrooms","overcrowding","tipovivi1","tipovivi2","tipovivi3","tipovivi4"]]
Ytrain2=train1.Target #0.4394
maxi,maxk=0,0
for k in range (6,30):
    knn = KNeighborsClassifier(n_neighbors=k,p=1)
    knn.fit(Xtrain2,Ytrain2)
    scores = cross_val_score(knn, Xtrain2,Ytrain2, cv=10)
    media=0
    for l in scores:
        media+=l
    media=media/10
    if media>maxi:
        maxi=media
        maxk=k
print(maxi,maxk)
knn = KNeighborsClassifier(n_neighbors=maxk,p=1)
knn.fit(Xtrain2,Ytrain2)
scores = cross_val_score(knn, Xtrain2,Ytrain2, cv=10)
scores
Xtest3 = Xtest2.fillna(-1)
Ytest3 = knn.predict(Xtest3)
result = np.vstack((testcosta["Id"], Ytest3)).T
sub = pd.DataFrame(columns = ["Id","Target"], data = result)
sub.to_csv("PMR3508-2018-7dd3037079.csv", index = False)
