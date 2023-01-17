import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import os 
os.listdir('../input')
dataTrain = pd.read_csv('./input/train.csv')
print(dataTrain.shape)
dataTrain.head()
XdataTrain = dataTrain.iloc[:, 1:-1]
YdataTrain = dataTrain['median_house_value']
IDdataTrain = dataTrain['Id']
dataTest = pd.read_csv('../input/test.csv')
print(dataTest.shape)
XdataTest = dataTest.iloc[:, 1:]
IDdataTest = dataTest['Id']
dataTest.head()
dataTrain.mean()[1:-1]
plot = dataTrain.mean()[4:-1].plot('bar')
plot = dataTrain.mean()[4:-2].plot('bar')
dataTrain.std()[1:-1]
plot = dataTrain.std()[4:-1].plot('bar')
plot = dataTrain.mean()[3:-2].plot('bar')
plot = dataTrain.std()[4:-2].plot('bar')
plot = plt.matshow(dataTrain.corr())
plot = dataTrain.corr().iloc[1:-1, -1].plot('bar')
plot = dataTrain.iloc[:, 3:-1].plot(kind='hist', bins=100, legend=True, alpha=0.3, xlim=(-1000, 70000))
plot = dataTrain.iloc[:, 3:6].plot(kind='hist', bins=100, legend=True, alpha=0.3, xlim=(-100, 6000))
plot = dataTrain.iloc[:, 6:-1].plot(kind='hist', bins=100, legend=True, alpha=0.3, xlim=(-1000, 70000))
import matplotlib.colors as mcolors
for i in list(XdataTrain):
    plt.hist2d(XdataTrain[i], YdataTrain, bins=100, norm=mcolors.PowerNorm(0.3))
    plt.title(i)
    plt.show()
plot = dataTrain['median_house_value'].plot(kind='hist', bins=100, alpha=0.5)
dataTrain['median_house_value'].max()
(dataTrain['median_house_value']==500001).sum()
dataTrain['median_house_value'].min()
(dataTrain['median_house_value']==14999).sum()
XbalTrain = XdataTrain.copy()
YbalTrain = YdataTrain.copy()
maxValor = (XdataTrain[(YdataTrain==500001)])
lenMax = len(XdataTrain[(YdataTrain==500001)])

np.random.seed(3508)

remove_n = int(4 * lenMax / 5)
drop_indices = np.random.choice(maxValor.index, remove_n, replace=False)
XbalTrain = XbalTrain.drop(drop_indices)
YbalTrain = YbalTrain.drop(drop_indices)

XbalTrain = XbalTrain.dropna()
YbalTrain = YbalTrain.dropna()

print(XbalTrain.shape)
print(YbalTrain.shape)
plot = YbalTrain.plot(kind='hist', bins=100, alpha=0.5)
X2grauTrain = XbalTrain.copy()
headers = list(XdataTrain)
for i in headers:
    for j in headers:
        if i==j:
            X2grauTrain[i+'^2'] = X2grauTrain[i] * X2grauTrain[j]
            break
        else:
            X2grauTrain[i+'x'+j] = X2grauTrain[i] * X2grauTrain[j]
X2grauTrain.head()
X3grauTrain = X2grauTrain.copy()
for i in headers:
    for j in headers:
            if i==j:
                for k in headers:
                    if k==j:
                        X2grauTrain[i+'^3'] = X3grauTrain[i] * X3grauTrain[j] * X3grauTrain[k]
                        break
                    else:
                        X2grauTrain[i+'^2x'+k] = X3grauTrain[i] * X3grauTrain[j] * X3grauTrain[k]
                break
            else:
                for k in headers:
                    if k==j:
                        X2grauTrain[i+'x'+j+'^2'] = X3grauTrain[i] * X3grauTrain[j] * X3grauTrain[k]
                        break
                    else:
                        X2grauTrain[i+'x'+j+'x'+k] = X3grauTrain[i] * X3grauTrain[j] * X3grauTrain[k]
X2grauTrain.head()
Xpos1Train = XbalTrain.copy()
Xpos1Train = Xpos1Train[['longitude', 'latitude']]
Xpos1Train.head()
Xpos2Train = Xpos1Train.copy()
headers = list(Xpos1Train)
for i in headers:
    for j in headers:
        if i==j:
            Xpos2Train[i+'^2'] = Xpos2Train[i] * Xpos2Train[j]
            break
        else:
            Xpos2Train[i+'x'+j] = Xpos2Train[i] * Xpos2Train[j]
Xpos2Train.head()
Xpos3Train = Xpos2Train.copy()
for i in headers:
    for j in headers:
            if i==j:
                for k in headers:
                    if k==j:
                        Xpos3Train[i+'^3'] = Xpos3Train[i] * Xpos3Train[j] * Xpos3Train[k]
                        break
                    else:
                        Xpos3Train[i+'^2x'+k] = Xpos3Train[i] * Xpos3Train[j] * Xpos3Train[k]
                break
            else:
                for k in headers:
                    if k==j:
                        Xpos3Train[i+'x'+j+'^2'] = Xpos3Train[i] * Xpos3Train[j] * Xpos3Train[k]
                        break
                    else:
                        Xpos3Train[i+'x'+j+'x'+k] = Xpos3Train[i] * Xpos3Train[j] * Xpos3Train[k]
Xpos3Train.head()
XnoPos1Train = XbalTrain.copy()
XnoPos1Train = XnoPos1Train.iloc[:, 2:]
XnoPos1Train.head()
XnoPos2Train = XnoPos1Train.copy()
headers = list(XnoPos1Train)
for i in headers:
    for j in headers:
        if i==j:
            XnoPos2Train[i+'^2'] = XnoPos2Train[i] * XnoPos2Train[j]
            break
        else:
            XnoPos2Train[i+'x'+j] = XnoPos2Train[i] * XnoPos2Train[j]
XnoPos2Train.head()
XnoPos3Train = XnoPos2Train.copy()
for i in headers:
    for j in headers:
            if i==j:
                for k in headers:
                    if k==j:
                        XnoPos3Train[i+'^3'] = XnoPos3Train[i] * XnoPos3Train[j] * XnoPos3Train[k]
                        break
                    else:
                        XnoPos3Train[i+'^2x'+k] = XnoPos3Train[i] * XnoPos3Train[j] * XnoPos3Train[k]
                break
            else:
                for k in headers:
                    if k==j:
                        XnoPos3Train[i+'x'+j+'^2'] = XnoPos3Train[i] * XnoPos3Train[j] * XnoPos3Train[k]
                        break
                    else:
                        XnoPos3Train[i+'x'+j+'x'+k] = XnoPos3Train[i] * XnoPos3Train[j] * XnoPos3Train[k]
XnoPos3Train.head()
from sklearn.metrics import make_scorer
def RMSLErelu(Y, Ypred):
    n = len(Y)
    soma = 0
    Y = np.array(Y)
    for i in range(len(Y)):
        #Caso Ypred[i] for negativo eu trato como se ele fosse 0
        if Ypred[i] > 0:
            soma += ( math.log(Ypred[i]+1) - math.log(Y[i]+1) )**2
        else:
            soma += math.log(Y[i]+1)**2
    return math.sqrt(soma / n)
scorerRelu = make_scorer(RMSLErelu)
def RMSLEabs(Y, Ypred):
    n = len(Y)
    soma = 0
    Y = np.array(Y)
    for i in range(len(Y)):
        soma += ( math.log( abs(Ypred[i]) + 1 ) - math.log( Y[i] + 1 ) )**2
    return math.sqrt(soma / n)
scorerAbs = make_scorer(RMSLEabs)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, XbalTrain,YbalTrain, cv=10, scoring=scorerRelu)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
from sklearn.linear_model import Lasso
clf = Lasso()
print(clf)
scores = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu)
print(scores)
print(scores.mean())
clf = Lasso()
print(clf)
scores = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
from sklearn.linear_model import Ridge
clf = Ridge()
print(clf)
scores = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu)
print(scores)
print(scores.mean())
clf = Ridge()
print(clf)
scores = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, X2grauTrain,YbalTrain, cv=10, scoring=scorerRelu)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, X2grauTrain,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
from sklearn.linear_model import LogisticRegression
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, X3grauTrain,YbalTrain, cv=10, scoring=scorerRelu)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, X3grauTrain,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, Xpos1Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, Xpos2Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, Xpos3Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, XnoPos1Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, XnoPos2Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
clf = LinearRegression()
print(clf)
scores = cross_val_score(clf, XnoPos3Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

accuracies = {}

Ptam = 5
Ktam = 25

accZ = np.empty([Ptam, Ktam])*0

for k in tqdm(range(1,Ktam+1)):
    for p in range(1,Ptam+1):
        knn = KNeighborsRegressor(n_neighbors=k, p=p)
        scores = cross_val_score(knn, XbalTrain, YbalTrain, cv=10, scoring=scorerAbs)
        
        accuracies[(k,p)] = scores.mean()
        accZ[p-1][k-1] = scores.mean()
accuraciesSorted = list(accuracies.items())
accuraciesSorted.sort(key=lambda x: x[1])

accuraciesSorted[:10]
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1, 3, 1, projection='3d')

accK = np.arange(1, Ktam+1)
accP = np.arange(1, Ptam+1)

accK, accP = np.meshgrid(accK, accP)

ax.plot_wireframe(accK, accP, accZ)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_wireframe(accK, accP, accZ)
ax.view_init(30, 30)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_wireframe(accK, accP, accZ)
ax.view_init(20,90)
accuracies = {}

for k in tqdm(range(1,100)):
    knn = KNeighborsRegressor(n_neighbors=k, p=1)
    scores = cross_val_score(knn, XbalTrain, YbalTrain, cv=10, scoring=scorerAbs)
    accuracies[k] = scores.mean()
accuraciesSorted = list(accuracies.items())
accuraciesSorted.sort(key=lambda x: x[1])

accuraciesSorted[:10]
accX = sorted(list(accuracies.keys()))
accY = [accuracies[i] for i in accX]
plot = plt.plot(accX, accY)
plot = plt.plot(accX[4:30], accY[4:30], 'bo', accX[4:30], accY[4:30])
Esq = np.polyfit(accX[4:10], accY[4:10], 1)
Dir = np.polyfit(accX[10:30], accY[10:30], 1)
print(round((Esq[1]-Dir[1])/(Dir[0]-Esq[0])))
knn = KNeighborsRegressor(n_neighbors=10, p=1)
print(knn)
scores = cross_val_score(knn, XbalTrain, YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
accuracies = {}

for k in tqdm(range(1,100)):
    knn = KNeighborsRegressor(n_neighbors=k, p=2)
    scores = cross_val_score(knn, Xpos1Train,YbalTrain, cv=10, scoring=scorerAbs)
    accuracies[k] = scores.mean()
accuraciesSorted = list(accuracies.items())
accuraciesSorted.sort(key=lambda x: x[1])

accuraciesSorted[:10]
accX = sorted(list(accuracies.keys()))
accY = [accuracies[i] for i in accX]
plot = plt.plot(accX, accY)
plot = plt.plot(accX[3:10], accY[3:10], 'bo', accX[3:10], accY[3:10])
knn = KNeighborsRegressor(n_neighbors=6, p=2)
print(knn)
scores = cross_val_score(knn, Xpos1Train,YbalTrain, cv=10, scoring=scorerAbs)
print(scores)
print(scores.mean())
from sklearn.tree import DecisionTreeRegressor
scores = {}
for i in tqdm(range(1,26)):
    clf = DecisionTreeRegressor(max_depth=i, random_state=3508)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()

clf = DecisionTreeRegressor(random_state=3508)
clf.fit(XdataTrain, YdataTrain)
scores[i+1] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
scrX = list(scores.keys())
scrY = list(scores.values())

plot = plt.plot(scrX, scrY, 'bo', scrX, scrY)
plot = plt.plot(scrX[7:13], scrY[7:13], 'bo', scrX[7:13], scrY[7:13])
scores = {}
for i in tqdm(range(2,102)):
    clf = DecisionTreeRegressor(max_depth=10, min_samples_split=i, random_state=3508)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
scrX = list(scores.keys())
scrY = list(scores.values())

plot = plt.plot(scrX, scrY, 'bo', scrX, scrY)
plot = plt.plot(scrX[10:90], scrY[10:90], 'bo', scrX[10:90], scrY[10:90])
scores = {}
for i in tqdm(range(1,51)):
    clf = DecisionTreeRegressor(max_depth=10, min_samples_split=50, min_samples_leaf=i, random_state=3508)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
scrX = list(scores.keys())
scrY = list(scores.values())

plot = plt.plot(scrX, scrY, 'bo', scrX, scrY)
plot = plt.plot(scrX[5:30], scrY[5:30], 'bo', scrX[5:30], scrY[5:30])
scores = {}
for i in tqdm(range(1,51)):
    clf = DecisionTreeRegressor(max_depth=i, min_samples_split=50, min_samples_leaf=15, random_state=3508)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()

clf = DecisionTreeRegressor(min_samples_split=50, min_samples_leaf=15, random_state=3508)
scores[i+1] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
    
scrX = list(scores.keys())
scrY = list(scores.values())

plot = plt.plot(scrX, scrY, 'bo', scrX, scrY)
plt.show()
plot = plt.plot(scrX[7:13], scrY[7:13], 'bo', scrX[7:13], scrY[7:13])
plt.show()
from sklearn.ensemble import BaggingRegressor
import time
'''scores = {}
temps = []
t = time.time()
for i in tqdm(range(1, 121, 10)):
    clf = BaggingRegressor(DecisionTreeRegressor(random_state=3508), random_state=1165842557, n_estimators=i)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
    temps.append(time.time()-t)
    t = time.time()

scrX = list(scores.keys())
scrY = list(scores.values())

plot = plt.plot(scrX, scrY, 'bo', scrX, scrY, scrX, temps)
plt.title('Score pela quantidade de árvores no Bagging')

plt.plot(scrX, temps, 'bo', scrX, temps)
plt.title('Tempo de treinamento pela quantidade \nde árvores no Bagging')'''
pass
clf = BaggingRegressor(DecisionTreeRegressor(random_state=3508), random_state=1165842557, n_estimators=80)
print(clf)
score = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu)
print(score)
print(score.mean())
scores = {}
temps = []
t = time.time()
for i in tqdm(range(1, 21)):
    clf = BaggingRegressor(DecisionTreeRegressor(max_depth=i, random_state=3508), random_state=1165842557, n_estimators=50)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
    temps.append(time.time()-t)
    t = time.time()
clf = BaggingRegressor(DecisionTreeRegressor(random_state=3508), random_state=1165842557, n_estimators=50)
scores[i+1] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
temps.append(time.time()-t)

scrX = list(scores.keys())
scrY = list(scores.values())

plt.plot(scrX, scrY, 'bo', scrX, scrY)
plt.show()
plt.plot(scrX, temps, 'bo', scrX, temps)
plt.show()
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=1165842557, )
print(clf)
score = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu)
print(score)
print(score.mean())
from sklearn.ensemble import GradientBoostingRegressor
scores = {}
temps = []
t = time.time()
for i in tqdm(range(50, 250, 10)):
    clf = GradientBoostingRegressor(max_depth=2, n_estimators=i)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
    temps.append(time.time()-t)
    t = time.time()
clf = GradientBoostingRegressor(max_depth=2)
scores[i+1] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
temps.append(time.time()-t)

scrX = list(scores.keys())
scrY = list(scores.values())

plt.plot(scrX, scrY, 'bo', scrX, scrY)
plt.show()
plt.plot(scrX, temps, 'bo', scrX, temps)
plt.show()
scores = {}
temps = []
t = time.time()
for i in tqdm(range(1, 11)):
    clf = GradientBoostingRegressor(max_depth=i, n_estimators=140)
    scores[i] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
    temps.append(time.time()-t)
    t = time.time()
clf = GradientBoostingRegressor(n_estimators=140)
scores[i+1] = cross_val_score(clf, XbalTrain, YbalTrain, cv=10, scoring=scorerRelu).mean()
temps.append(time.time()-t)

scrX = list(scores.keys())
scrY = list(scores.values())

plt.plot(scrX, scrY, 'bo', scrX, scrY)
plt.show()
plt.plot(scrX, temps, 'bo', scrX, temps)
plt.show()
clf = GradientBoostingRegressor(max_depth=8, n_estimators=140)
clf.fit(Xpos1Train,YbalTrain)
YdataPred = clf.predict(XdataTest.iloc[:, :2])
ID = list(IDdataTest)
YdataPred = list(YdataPred)
submission = np.array([ID, YdataPred])

submission = pd.DataFrame(submission.T, columns=['Id', 'median_house_value'])
submission['Id'] = submission['Id'].astype(int)
submission.to_csv('out.csv', index=False)