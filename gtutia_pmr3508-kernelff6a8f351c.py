%matplotlib inline

import matplotlib
import pandas as pd
house = pd.read_csv("../input/treino-costa-rica/train.csv",na_values='NaN')
house.shape
house.head()
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150
house.isnull().sum()
house = house.drop(['v2a1','v18q1','rez_esc'],axis=1)
house_full = house.dropna()
house_full.shape
house_full.head()
house_full_num = house_full.drop(['Id','idhogar','dependency','edjefe','edjefa'],axis=1)

house_full_num.shape
Xhouse = house_full_num.drop(["Target"],axis=1)
Yhouse = house_full_num.Target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,Xhouse,Yhouse,cv=10)
scores
scores.mean()
knn2 = KNeighborsClassifier(n_neighbors=83)
scores2 = cross_val_score(knn2,Xhouse,Yhouse,cv=10)
scores2.mean()
Xhouse2 = house_full_num.drop(["Target","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq"],axis=1)
Yhouse2 = house_full_num.Target

from sklearn.model_selection import cross_val_score
scores3 = cross_val_score(knn2,Xhouse2,Yhouse2,cv=10)
scores3.mean()
Xhouse3 = house_full_num.drop(["Target","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq","paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother"],axis=1)
Yhouse3 = house_full_num.Target

from sklearn.model_selection import cross_val_score
scores4 = cross_val_score(knn2,Xhouse3,Yhouse3,cv=10)
scores4.mean()
Xhouse3 = house_full_num.drop(["Target","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq","paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother","pisomoscer","pisocemento","pisoother","pisonatur","pisomadera"],axis=1)
Yhouse3 = house_full_num.Target

from sklearn.model_selection import cross_val_score
scores4 = cross_val_score(knn2,Xhouse3,Yhouse3,cv=10)
scores4.mean()
Xhouse4 = house_full_num.drop(["Target","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq","paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother","pisomoscer","pisocemento","pisoother","pisonatur","pisomadera","techozinc","techoentrepiso","techocane","techootro"],axis=1)
Yhouse4 = house_full_num.Target

from sklearn.model_selection import cross_val_score
scores5 = cross_val_score(knn2,Xhouse4,Yhouse4,cv=10)
scores5.mean()
Xhouse5 = house_full_num.drop(["Target","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq","paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother","pisomoscer","pisocemento","pisoother","pisonatur","pisomadera","epared1","epared2","epared3","etecho1","etecho2","etecho3","eviv1","eviv2","eviv3"],axis=1)
Yhouse5 = house_full_num.Target

from sklearn.model_selection import cross_val_score
scores6 = cross_val_score(knn2,Xhouse5,Yhouse5,cv=10)
scores6.mean()
Xhouse6 = house_full_num.drop(["Target","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq","paredblolad","paredzocalo","paredpreb","pareddes","paredmad","paredzinc","paredfibras","paredother","pisomoscer","pisocemento","pisoother","pisonatur","pisomadera","epared1","epared2","epared3","etecho1","etecho2","etecho3","eviv1","eviv2","eviv3","hacdor","hacapo"],axis=1)
Yhouse6 = house_full_num.Target

from sklearn.model_selection import cross_val_score
scores7 = cross_val_score(knn2,Xhouse6,Yhouse6,cv=10)
scores7.mean()
knn3 = KNeighborsClassifier(n_neighbors=90)
scores10 = cross_val_score(knn3,Xhouse6,Yhouse6,cv=10)
scores10.mean()