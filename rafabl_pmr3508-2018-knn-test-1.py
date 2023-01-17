import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
CostaRica = pd.read_csv('../input/train.csv', index_col = 'Id')
CostaRica = pd.read_csv('../input/train.csv', index_col = 'Id')
CostaRica

plt.hist(CostaRica['Target'])
#1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households
# That the more than half of the test data is from non vunerable households
plt.hist(CostaRica['age'])

plt.hist(CostaRica['escolari'])
#more than half (4910, with a 9557 population) of the people have escolarity equal or less than 6 years
# First test : Using knn with only construction variables
TestCostaRica = pd.read_csv('../input/test.csv', index_col = 'Id')
TestCostaRica.head()

TestCostaRica

XCostaRica = CostaRica[["hhsize","paredblolad", "paredzocalo", "paredpreb","pareddes", "paredmad", "paredzinc", "paredfibras", "paredother","pisomoscer", "pisocemento","pisoother", "pisonatur", "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techocane","techootro", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele", "sanitario1","sanitario2", "sanitario3", "sanitario5", "sanitario6", "energcocinar1","energcocinar2","energcocinar3","energcocinar4", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6","epared1", "epared2","epared3","etecho1","etecho2","etecho3","eviv1","eviv2","eviv3"]]
XCostaRica
X1CostaRica = CostaRica[["hhsize","paredblolad", "paredzocalo", "paredpreb","pareddes", "paredmad", "paredzinc", "paredfibras", "paredother","pisomoscer", "pisocemento","pisoother", "pisonatur", "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techocane","techootro", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele", "sanitario1","sanitario2", "sanitario3", "sanitario5", "sanitario6", "energcocinar1","energcocinar2","energcocinar3","energcocinar4", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6","epared1", "epared2","epared3","etecho1","etecho2","etecho3","eviv1","eviv2","eviv3","tipovivi1","tipovivi2","tipovivi3","tipovivi4","tipovivi5"]]
X2CostaRica = CostaRica[["refrig","v18q","v18q1","escolari","instlevel1","instlevel2","instlevel3","instlevel4","instlevel5","instlevel6","instlevel7","instlevel8","instlevel9","computer"]]

YCostaRica = CostaRica.Target
YCostaRica
XTestCostaRica = TestCostaRica[["hhsize","paredblolad", "paredzocalo", "paredpreb","pareddes", "paredmad", "paredzinc", "paredfibras", "paredother","pisomoscer", "pisocemento","pisoother", "pisonatur", "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techocane","techootro", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele", "sanitario1","sanitario2", "sanitario3", "sanitario5", "sanitario6", "energcocinar1","energcocinar2","energcocinar3","energcocinar4", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6","epared1", "epared2","epared3","etecho1","etecho2","etecho3","eviv1","eviv2","eviv3"]]
X1TestCostaRica = TestCostaRica[["hhsize","paredblolad", "paredzocalo", "paredpreb","pareddes", "paredmad", "paredzinc", "paredfibras", "paredother","pisomoscer", "pisocemento","pisoother", "pisonatur", "pisonotiene", "pisomadera", "techozinc", "techoentrepiso", "techocane","techootro", "cielorazo", "abastaguadentro", "abastaguafuera", "abastaguano", "public", "planpri", "noelec", "coopele", "sanitario1","sanitario2", "sanitario3", "sanitario5", "sanitario6", "energcocinar1","energcocinar2","energcocinar3","energcocinar4", "elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu5", "elimbasu6","epared1", "epared2","epared3","etecho1","etecho2","etecho3","eviv1","eviv2","eviv3","tipovivi1","tipovivi2","tipovivi3","tipovivi4","tipovivi5"]]
X2TestCostaRica = TestCostaRica[["refrig","v18q","v18q1","escolari","instlevel1","instlevel2","instlevel3","instlevel4","instlevel5","instlevel6","instlevel7","instlevel8","instlevel9","computer"]]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 7)
knn1= KNeighborsClassifier(n_neighbors = 7)


from sklearn.model_selection import cross_val_score


scores = cross_val_score(knn, XCostaRica, YCostaRica, cv=10)
scores

scores1 = cross_val_score(knn, X1CostaRica, YCostaRica, cv=10)
scores1 
# Por conter muitos NaN, não fica Viável scores2 = cross_val_score(knn, X2CostaRica, YCostaRica, cv=10)
# scores2 
knn.fit(XCostaRica, YCostaRica)

knn1.fit(X1CostaRica, YCostaRica)

YTestPredict = knn.predict(XTestCostaRica)
YTestPredict1 = knn1.predict(X1TestCostaRica)
YTestPredict 
YTestPredict1
YTestPredict.shape

YTestPredict1.shape

Predict = pd.DataFrame (index = TestCostaRica.index)
Predict1 = pd.DataFrame (index = TestCostaRica.index)
Predict['Target'] = YTestPredict
Predict1['Target'] = YTestPredict1
Predict

Predict1
#Segundo teste com adição da situação da casa
Predict.to_csv("Prediction1.csv")
# Segundo Teste: Adicionando colunas da situação das casas
Predict1.to_csv("Prediction2.csv")
