import pandas as pd
import sklearn

adult = pd.read_csv("../input/train.csv",
        names=[
        'Id','v2a1','hacdor','rooms','hacapo','v14a','refrig','v18q','v18q1','r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3',
        'tamhog','tamviv','escolari','rez_esc','hhsize','paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc',
        'paredfibras','paredother','pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera','techozinc',
        'techoentrepiso','techocane','techootro','cielorazo','abastaguadentro','abastaguafuera','abastaguano','elec1','elec2','elec3',
        'elec4','sanitario1','sanitario2','sanitario3','sanitario5','sanitario6','energcocinar1','energcocinar2','energcocinar3',
        'energcocinar4','elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6','epared1','epared2','epared3','etecho1',
        'etecho2','etecho3','eviv1','eviv2','eviv3','dis','male','female','estadocivil1','estadocivil2','estadocivil3','estadocivil4',
        'estadocivil5','estadocivil6','estadocivil7','parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6',
        'parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12','idhogar','hogar_nin','hogar_adul',
        'hogar_mayor','hogar_total','dependency','edjefe','edjefa','meaneduc','instlevel1','instlevel2','instlevel3','instlevel4','instlevel5',
        'instlevel6','instlevel7','instlevel8','instlevel9','bedrooms','overcrowding','tipovivi1','tipovivi2','tipovivi3','tipovivi4',
        'tipovivi5','computer','television','mobilephone','qmobilephone','lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','area1',
        'area2','age','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned',
        'agesq','Target'],
        sep=r'\s*,\s*',
        engine='python',
        na_values=None)
adult = adult.drop(['overcrowding'],axis=1)
adult = adult.drop(['hacdor','hacapo'],axis=1)
adult = adult.drop(['tamhog'],axis=1)
adult = adult.drop(['Id','idhogar','dependency','hogar_nin','hogar_adul','hogar_total','hhsize','tamviv','v18q',
                    'r4h3','r4m3','r4t1','r4t2','r4t3','mobilephone','SQBescolari','SQBage','SQBhogar_total',
                    'SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned',
        'agesq'],axis=1)
adult.insert(2, 'Gender', 0)
for i in range(1,len(adult['Gender'])):
    if adult.loc[i,('female')] == '1':
        adult.loc[i,('Gender')] = 1

adult = adult.drop(['male','female'], axis=1)
def agrupa(dat,st,st2):
    dat.insert(2, st, 0)
    for i in range(1,len(dat[st])):
        for j in range(len(st2)):
            m=st+str(st2[j])
            if dat.loc[i,(m)] == '1':
                dat.loc[i,(st)] = j
    for x in range(len(st2)):
        k=st+str(st2[x])
        dat = dat.drop([k],axis=1)
    return dat
adult = agrupa(adult,'estadocivil',range(1,8))
adult = agrupa(adult,'lugar',range(1,7))
adult = agrupa(adult,'area',range(1,3))
adult = agrupa(adult,'tipovivi',range(1,6))
adult = agrupa(adult,'instlevel',range(1,10))
adult = agrupa(adult,'parentesco',range(1,13))
adult = agrupa(adult,'eviv',range(1,4))
adult = agrupa(adult,'etecho',range(1,4))
adult = agrupa(adult,'epared',range(1,4))
adult = agrupa(adult,'elimbasu',range(1,7))
adult = agrupa(adult,'energcocinar', range(1,5))
adult = agrupa(adult,'sanitario', [1,2,3,5,6])
adult = agrupa(adult,'pared',['blolad','zocalo','preb','des','mad','zinc','fibras','other'])
adult = agrupa(adult,'piso',['moscer','cemento','other','natur','notiene','madera'])
adult = agrupa(adult,'techo',['zinc','entrepiso','cane','otro'])
adult = agrupa(adult,'abastagua',['no','dentro','fuera'])
adult = agrupa(adult,'elec',range(1,5))
adult.insert(2, 'edjef', 0)
for i in range(1,len(adult['edjef'])):
    if adult.loc[i,('edjefe')] != 'no' and adult.loc[i,('edjefe')] != 'yes':
        adult.loc[i,('edjef')] = adult.loc[i,('edjefe')]
    elif adult.loc[i,('edjefa')] != 'no' and adult.loc[i,('edjefa')] != 'yes':
        adult.loc[i,('edjef')] = adult.loc[i,('edjefa')]
adult = adult.drop(['edjefe','edjefa'],axis=1)
adult=adult.drop(adult.index[[0]])
list(adult)
adult[['v2a1',
 'rooms',
 'edjef',
 'elec',
 'abastagua',
 'techo',
 'piso',
 'pared',
 'sanitario',
 'energcocinar',
 'elimbasu',
 'epared',
 'etecho',
 'eviv',
 'parentesco',
 'instlevel',
 'tipovivi',
 'area',
 'lugar',
 'estadocivil',
 'Gender',
 'v14a',
 'refrig',
 'v18q1',
 'r4h1',
 'r4h2',
 'r4m1',
 'r4m2',
 'escolari',
 'rez_esc',
 'cielorazo',
 'dis',
 'hogar_mayor',
 'meaneduc',
 'bedrooms',
 'computer',
 'television',
 'qmobilephone',
 'age',
 ]] = adult[['v2a1',
 'rooms',
 'edjef',
 'elec',
 'abastagua',
 'techo',
 'piso',
 'pared',
 'sanitario',
 'energcocinar',
 'elimbasu',
 'epared',
 'etecho',
 'eviv',
 'parentesco',
 'instlevel',
 'tipovivi',
 'area',
 'lugar',
 'estadocivil',
 'Gender',
 'v14a',
 'refrig',
 'v18q1',
 'r4h1',
 'r4h2',
 'r4m1',
 'r4m2',
 'escolari',
 'rez_esc',
 'cielorazo',
 'dis',
 'hogar_mayor',
 'meaneduc',
 'bedrooms',
 'computer',
 'television',
 'qmobilephone',
 'age',
 ]].astype(float)
import random
adult['Gender'].fillna(random.randint(0,1), inplace=True)
import numpy as np
adult[['rooms','v2a1','meaneduc','age','r4h1','r4h2','r4m1','r4m2']] = adult[['rooms',
                                'v2a1','meaneduc','age','r4h1','r4h2','r4m1','r4m2']].fillna(adult.mean())

adult['v2a1'].mean()
adult['v2a1'] = adult['v2a1'].fillna(adult['v2a1'].mean())
nam=['edjef','elec','abastagua','techo','piso','pared','sanitario','energcocinar','elimbasu','epared',
       'etecho','eviv','parentesco','instlevel','tipovivi','area','lugar','estadocivil','v14a','refrig','v18q1',
       'escolari','rez_esc','cielorazo','dis','hogar_mayor','bedrooms','computer','television','qmobilephone']
for f in nam:
    adult[f]=adult[f].fillna(int(adult[f].mean()))
nadult = adult.dropna()
nadult
test = pd.read_csv("../input/test.csv",
        names=[
        'Id','v2a1','hacdor','rooms','hacapo','v14a','refrig','v18q','v18q1','r4h1','r4h2','r4h3','r4m1','r4m2','r4m3','r4t1','r4t2','r4t3',
        'tamhog','tamviv','escolari','rez_esc','hhsize','paredblolad','paredzocalo','paredpreb','pareddes','paredmad','paredzinc',
        'paredfibras','paredother','pisomoscer','pisocemento','pisoother','pisonatur','pisonotiene','pisomadera','techozinc',
        'techoentrepiso','techocane','techootro','cielorazo','abastaguadentro','abastaguafuera','abastaguano','elec1','elec2','elec3',
        'elec4','sanitario1','sanitario2','sanitario3','sanitario5','sanitario6','energcocinar1','energcocinar2','energcocinar3',
        'energcocinar4','elimbasu1','elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6','epared1','epared2','epared3','etecho1',
        'etecho2','etecho3','eviv1','eviv2','eviv3','dis','male','female','estadocivil1','estadocivil2','estadocivil3','estadocivil4',
        'estadocivil5','estadocivil6','estadocivil7','parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6',
        'parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12','idhogar','hogar_nin','hogar_adul',
        'hogar_mayor','hogar_total','dependency','edjefe','edjefa','meaneduc','instlevel1','instlevel2','instlevel3','instlevel4','instlevel5',
        'instlevel6','instlevel7','instlevel8','instlevel9','bedrooms','overcrowding','tipovivi1','tipovivi2','tipovivi3','tipovivi4',
        'tipovivi5','computer','television','mobilephone','qmobilephone','lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','area1',
        'area2','age','SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned',
        'agesq'],
        sep=r'\s*,\s*',
        engine='python',
        na_values=None)


test = test.drop(['overcrowding'],axis=1)
test = test.drop(['hacdor','hacapo'],axis=1)
test = test.drop(['tamhog'],axis=1)
test = test.drop(['Id','idhogar','dependency','hogar_nin','hogar_adul','hogar_total','hhsize','tamviv','v18q',
                    'r4h3','r4m3','r4t1','r4t2','r4t3','mobilephone','SQBescolari','SQBage','SQBhogar_total',
                    'SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned',
        'agesq'],axis=1)
test.insert(2, 'Gender', 0)
for i in range(1,len(test['Gender'])):
    if test.loc[i,('female')] == '1':
        test.loc[i,('Gender')] = 1

test = test.drop(['male','female'], axis=1)
test = agrupa(test,'estadocivil',range(1,8))
test = agrupa(test,'lugar',range(1,7))
test = agrupa(test,'area',range(1,3))
test = agrupa(test,'tipovivi',range(1,6))
test = agrupa(test,'instlevel',range(1,10))
test = agrupa(test,'parentesco',range(1,13))
test = agrupa(test,'eviv',range(1,4))
test = agrupa(test,'etecho',range(1,4))
test = agrupa(test,'epared',range(1,4))
test = agrupa(test,'elimbasu',range(1,7))
test = agrupa(test,'energcocinar', range(1,5))
test = agrupa(test,'sanitario', [1,2,3,5,6])
test = agrupa(test,'pared',['blolad','zocalo','preb','des','mad','zinc','fibras','other'])
test = agrupa(test,'piso',['moscer','cemento','other','natur','notiene','madera'])
test = agrupa(test,'techo',['zinc','entrepiso','cane','otro'])
test = agrupa(test,'abastagua',['no','dentro','fuera'])
test = agrupa(test,'elec',range(1,5))
test.insert(2, 'edjef', 0)
for i in range(1,len(test['edjef'])):
    if test.loc[i,('edjefe')] != 'no' and test.loc[i,('edjefe')] != 'yes':
        test.loc[i,('edjef')] = test.loc[i,('edjefe')]
    elif test.loc[i,('edjefa')] != 'no' and test.loc[i,('edjefa')] != 'yes':
        test.loc[i,('edjef')] = test.loc[i,('edjefa')]
test = test.drop(['edjefe','edjefa'],axis=1)
test=test.drop(test.index[[0]])
test[['v2a1',
 'rooms',
 'edjef',
 'elec',
 'abastagua',
 'techo',
 'piso',
 'pared',
 'sanitario',
 'energcocinar',
 'elimbasu',
 'epared',
 'etecho',
 'eviv',
 'parentesco',
 'instlevel',
 'tipovivi',
 'area',
 'lugar',
 'estadocivil',
 'Gender',
 'v14a',
 'refrig',
 'v18q1',
 'r4h1',
 'r4h2',
 'r4m1',
 'r4m2',
 'escolari',
 'rez_esc',
 'cielorazo',
 'dis',
 'hogar_mayor',
 'meaneduc',
 'bedrooms',
 'computer',
 'television',
 'qmobilephone',
 'age', 
 ]] = test[['v2a1',
 'rooms',
 'edjef',
 'elec',
 'abastagua',
 'techo',
 'piso',
 'pared',
 'sanitario',
 'energcocinar',
 'elimbasu',
 'epared',
 'etecho',
 'eviv',
 'parentesco',
 'instlevel',
 'tipovivi',
 'area',
 'lugar',
 'estadocivil',
 'Gender',
 'v14a',
 'refrig',
 'v18q1',
 'r4h1',
 'r4h2',
 'r4m1',
 'r4m2',
 'escolari',
 'rez_esc',
 'cielorazo',
 'dis',
 'hogar_mayor',
 'meaneduc',
 'bedrooms',
 'computer',
 'television',
 'qmobilephone',
 'age', 
 ]].astype(float)
test['Gender'].fillna(random.randint(0,1), inplace=True)
test[['rooms','v2a1','meaneduc','age','r4h1','r4h2','r4m1','r4m2']] = test[['rooms',
                                'v2a1','meaneduc','age','r4h1','r4h2','r4m1','r4m2']].fillna(test.mean())
test['v2a1'] = test['v2a1'].fillna(test['v2a1'].mean())
nam=['edjef','elec','abastagua','techo','piso','pared','sanitario','energcocinar','elimbasu','epared',
       'etecho','eviv','parentesco','instlevel','tipovivi','area','lugar','estadocivil','v14a','refrig','v18q1',
       'escolari','rez_esc','cielorazo','dis','hogar_mayor','bedrooms','computer','television','qmobilephone']
for f in nam:
    test[f]=test[f].fillna(test[f].mean())
ntest = test.dropna()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
Xadult = nadult[['v2a1', 'rooms','edjef','elec','abastagua','techo','piso','pared','sanitario','energcocinar','elimbasu','epared',
 'etecho','eviv','parentesco','instlevel','tipovivi','area','lugar','estadocivil','Gender','v14a','refrig','v18q1','r4h1','r4h2',
 'r4m1','r4m2','escolari','rez_esc','cielorazo','dis','hogar_mayor','meaneduc', 'bedrooms', 'computer', 'television', 'qmobilephone',
 'age']]
XtestAdult = ntest[['v2a1', 'rooms','edjef','elec','abastagua','techo','piso','pared','sanitario','energcocinar','elimbasu','epared',
 'etecho','eviv','parentesco','instlevel','tipovivi','area','lugar','estadocivil','Gender','v14a','refrig','v18q1','r4h1','r4h2',
 'r4m1','r4m2','escolari','rez_esc','cielorazo','dis','hogar_mayor','meaneduc', 'bedrooms', 'computer', 'television', 'qmobilephone',
 'age']]
Yadult = nadult.Target
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(Xadult,Yadult)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
YtestPred = knn.predict(XtestAdult)
YtestPred
