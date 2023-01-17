import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statistics import mode
trainData = pd.read_csv("../input/train.csv", header=0, index_col=0, na_values="?")
testData = pd.read_csv("../input/test.csv", header=0, index_col=0, na_values="?")
trainData.shape
trainData["lugar"] = 0
lugares = {"lugar1":1,"lugar2":2,"lugar3":3,"lugar4":4,"lugar5":5,"lugar6":6}

for key,value in lugares.items():
    trainData.loc[trainData[key]==1,"lugar"] = value
    testData.loc[testData[key]==1,"lugar"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["lugar"].value_counts().plot(kind="pie")
trainData["tipovivi"] = 0
tipovivi = {"tipovivi1":1,"tipovivi2":2,"tipovivi3":3,"tipovivi4":4,"tipovivi5":5}

for key,value in tipovivi.items():
    trainData.loc[trainData[key]==1,"tipovivi"] = value
    testData.loc[testData[key]==1,"tipovivi"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["tipovivi"].value_counts().plot(kind="pie")
trainData["instlevel"] = 0
instlevel = {"instlevel1":1,"instlevel2":2,"instlevel3":3,"instlevel4":4,"instlevel5":5,"instlevel6":6,"instlevel7":7,"instlevel8":8,"instlevel9":9}

for key,value in instlevel.items():
    trainData.loc[trainData[key]==1,"instlevel"] = value
    testData.loc[testData[key]==1,"instlevel"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["instlevel"].value_counts().plot(kind="bar")
trainData["parede"] = 0
parede = {"paredblolad":1,"paredzocalo":2,"paredpreb":3,"pareddes":4,"paredmad":5,"paredzinc":6,"paredfibras":7,"paredother":8}

for key,value in parede.items():
    trainData.loc[trainData[key]==1,"parede"] = value
    testData.loc[testData[key]==1,"parede"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["parede"].value_counts().plot(kind="bar")
trainData["piso"] = 0
piso = {"pisomoscer":1,"pisocemento":2,"pisoother":3,"pisonatur":4,"pisonotiene":5,"pisomadera":6}

for key,value in piso.items():
    trainData.loc[trainData[key]==1,"piso"] = value
    testData.loc[testData[key]==1,"piso"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["piso"].value_counts().plot(kind="bar")
trainData["techo"] = 0
techo = {"techozinc":1,"techoentrepiso":2,"techocane":3,"techootro":4}

for key,value in techo.items():
    trainData.loc[trainData[key]==1,"techo"] = value
    testData.loc[testData[key]==1,"techo"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["techo"].value_counts().plot(kind="bar")
trainData["abastagua"] = 0
abastagua = {"abastaguadentro":1,"abastaguafuera":2,"abastaguano":3}

for key,value in abastagua.items():
    trainData.loc[trainData[key]==1,"abastagua"] = value
    testData.loc[testData[key]==1,"abastagua"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["abastagua"].value_counts().plot(kind="bar")
trainData["elec"] = 0
elec = {"public":1,"planpri":2,"noelec":3, "coopele":4}

for key,value in elec.items():
    trainData.loc[trainData[key]==1,"elec"] = value
    testData.loc[testData[key]==1,"elec"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["elec"].value_counts().plot(kind="bar")
trainData["sanitario"] = 0
sanitario = {"sanitario1":1,"sanitario2":2,"sanitario3":3,"sanitario5":5,"sanitario6":6}

for key,value in sanitario.items():
    trainData.loc[trainData[key]==1,"sanitario"] = value
    testData.loc[testData[key]==1,"sanitario"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["sanitario"].value_counts().plot(kind="bar")
trainData["energcocinar"] = 0
energcocinar = {"energcocinar1":1,"energcocinar2":2,"energcocinar3":3,"energcocinar4":4}

for key,value in energcocinar.items():
    trainData.loc[trainData[key]==1,"energcocinar"] = value
    testData.loc[testData[key]==1,"energcocinar"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["energcocinar"].value_counts().plot(kind="bar")
trainData["elimbasu"] = 0
elimbasu = {"elimbasu1":1,"elimbasu2":2,"elimbasu3":3,"elimbasu4":4,"elimbasu5":5,"elimbasu6":6}

for key,value in elimbasu.items():
    trainData.loc[trainData[key]==1,"elimbasu"] = value
    testData.loc[testData[key]==1,"elimbasu"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["elimbasu"].value_counts().plot(kind="bar")
trainData["epared"] = 0
epared = {"epared1":1,"epared2":2,"epared3":3}

for key,value in epared.items():
    trainData.loc[trainData[key]==1,"epared"] = value
    testData.loc[testData[key]==1,"epared"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["epared"].value_counts().plot(kind="pie")
trainData["etecho"] = 0
etecho = {"etecho1":1,"etecho2":2,"etecho3":3}

for key,value in etecho.items():
    trainData.loc[trainData[key]==1,"etecho"] = value
    testData.loc[testData[key]==1,"etecho"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["etecho"].value_counts().plot(kind="pie")
trainData["eviv"] = 0
eviv = {"eviv1":1,"eviv2":2,"eviv3":3}

for key,value in eviv.items():
    trainData.loc[trainData[key]==1,"eviv"] = value
    testData.loc[testData[key]==1,"eviv"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["eviv"].value_counts().plot(kind="pie")
trainData["sex"] = 0
sex = {"male":1,"female":2}

for key,value in sex.items():
    trainData.loc[trainData[key]==1,"sex"] = value
    testData.loc[testData[key]==1,"sex"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["sex"].value_counts().plot(kind="bar")
trainData["estadocivil"] = 0
estadocivil = {"estadocivil1":1,"estadocivil2":2,"estadocivil3":3,"estadocivil4":4,"estadocivil5":5,"estadocivil6":6,"estadocivil7":7}

for key,value in estadocivil.items():
    trainData.loc[trainData[key]==1,"estadocivil"] = value
    testData.loc[testData[key]==1,"estadocivil"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["estadocivil"].value_counts().plot(kind="pie")
trainData["parentesco"] = 0
parentesco = {"parentesco1":1,"parentesco2":2,"parentesco3":3,"parentesco4":4,"parentesco5":5,"parentesco6":6,"parentesco7":7,"parentesco8":8,"parentesco9":9,"parentesco10":10,"parentesco11":11,"parentesco12":12}

for key,value in parentesco.items():
    trainData.loc[trainData[key]==1,"parentesco"] = value
    testData.loc[testData[key]==1,"parentesco"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["parentesco"].value_counts().plot(kind="bar")
trainData["area"] = 0
area = {"area1":1,"area2":2}

for key,value in area.items():
    trainData.loc[trainData[key]==1,"area"] = value
    testData.loc[testData[key]==1,"area"] = value
    trainData.drop(columns=key, inplace=True)
    testData.drop(columns=key, inplace=True)
trainData["area"].value_counts().plot(kind="pie")
trainData.head()
columnsToDrop = ['idhogar']

for i in range(len(trainData.columns)):
    if trainData.iloc[:,i].isnull().sum() > 1000:
        columnsToDrop.append(trainData.columns[i])
        
columnsToDrop
for i in range(len(columnsToDrop)):
    trainData.drop(columnsToDrop[i], axis=1, inplace=True)
    testData.drop(columnsToDrop[i], axis=1, inplace=True)
trainData.dropna(inplace=True)
testData.dropna(inplace=True)
trainData.loc[trainData["edjefe"]=='no' ,"edjefe"] = '0'
trainData.loc[trainData["edjefe"]=='yes',"edjefe"] = '1'
trainData.loc[trainData["edjefa"]=='no' ,"edjefa"] = '0'
trainData.loc[trainData["edjefa"]=='yes',"edjefa"] = '1'
trainData.loc[trainData["dependency"]=='no' ,"dependency"] = '0'
trainData.loc[trainData["dependency"]=='yes',"dependency"] = '1'

testData.loc[testData["edjefe"]=='no' ,"edjefe"] = '0'
testData.loc[testData["edjefe"]=='yes',"edjefe"] = '1'
testData.loc[testData["edjefa"]=='no' ,"edjefa"] = '0'
testData.loc[testData["edjefa"]=='yes',"edjefa"] = '1'
testData.loc[testData["dependency"]=='no' ,"dependency"] = '0'
testData.loc[testData["dependency"]=='yes',"dependency"] = '1'
trainData.head()
minmaxscaler = MinMaxScaler()

trainDataX = trainData.drop(columns="Target", inplace=False)
trainDataX = minmaxscaler.fit_transform(trainDataX)

testDataX  = testData
testDataX  = minmaxscaler.transform(testDataX)

trainDataY = trainData["Target"]
meanScoreManhattan = np.zeros(50)
stdScoreManhattan  = np.zeros(50)
for k in range(1,51):
    classifier = KNeighborsClassifier(n_neighbors=k, p=1)
    score = cross_val_score(classifier, trainDataX, trainDataY, cv=10)
    meanScoreManhattan[k-1] = np.mean(score)
    stdScoreManhattan[k-1]  = np.std(score)
    
np.amax(meanScoreManhattan)
meanScoreEuclidean = np.zeros(50)
stdScoreEuclidean  = np.zeros(50)
for k in range(1,51):
    classifier = KNeighborsClassifier(n_neighbors=k, p=2)
    score = cross_val_score(classifier, trainDataX, trainDataY, cv=10)
    meanScoreEuclidean[k-1] = np.mean(score)
    stdScoreEuclidean[k-1]  = np.std(score)
    
np.amax(meanScoreEuclidean)
if np.amax(meanScoreManhattan) > np.amax(meanScoreEuclidean):
    chosenK = np.argmax(meanScoreManhattan)+1
    chosenP = 1
else:
    chosenK = np.argmax(meanScoreEuclidean)+1
    chosenP = 2
    
chosenK
chosenP
plt.errorbar(range(1,51), meanScoreManhattan, yerr=1.96*np.array(stdScoreManhattan), fmt='-o')
plt.errorbar(range(1,51), meanScoreEuclidean, yerr=1.96*np.array(stdScoreEuclidean), fmt='-o')
classifier = KNeighborsClassifier(n_neighbors=chosenK,p=chosenP)
classifier.fit(trainDataX,trainDataY)
predictedData = classifier.predict(testDataX)
predictedData
output = pd.DataFrame(testData.index)
output["Target"] = predictedData
output
output.to_csv("PMR3508_MarcusPavani_HouseholdIncome.csv", index=False)