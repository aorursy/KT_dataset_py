import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

dataTrain = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
dataTrain = dataTrain[['rooms', 'v18q1', 'r4h3', 'r4m3', 'r4t3', 'tamhog', 'cielorazo', 'abastaguano',
                       'sanitario1', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3',
                       'eviv1', 'eviv2', 'eviv3', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4',
                       'estadocivil5', 'estadocivil6', 'estadocivil7', 'hogar_nin', 'hogar_adul', 'hogar_mayor',
                       'meaneduc', 'computer', 'television', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3',
                       'lugar4', 'lugar5', 'lugar6', 'area1', 'Target']]
dataTrain.head()
dataTrain.isnull().sum()[dataTrain.isnull().sum() > 0]
dataTrain = dataTrain.fillna(0)
dataTrain.head()
dataTrain['epared1'].add(2 * dataTrain['epared2'])
dataTrain['epared1'].add(3 * dataTrain['epared3'])

del dataTrain['epared2']
del dataTrain['epared3']
dataTrain['etecho1'].add(2 * dataTrain['etecho2'])
dataTrain['etecho1'].add(3 * dataTrain['etecho3'])

del dataTrain['etecho2']
del dataTrain['etecho3']
dataTrain['eviv1'].add(2 * dataTrain['eviv2'])
dataTrain['eviv1'].add(3 * dataTrain['eviv3'])

del dataTrain['eviv2']
del dataTrain['eviv3']
for i in range(2,8):
    dataTrain['estadocivil1'].add(i * dataTrain['estadocivil'+str(i)])
    del dataTrain['estadocivil'+str(i)]
for i in range(2,7):
    dataTrain['lugar1'].add(i * dataTrain['lugar'+str(i)])
    del dataTrain['lugar'+str(i)]
dataTrain['Target'].value_counts().plot('bar')
dataTarget4 = dataTrain.loc[dataTrain['Target'] == 4]
lenTarget4 = dataTarget4.shape[0]
np.random.seed(6)

remove_n = int(2 * lenTarget4 / 3)
drop_indices = np.random.choice(dataTarget4.index, remove_n, replace=False)
df_subset = dataTarget4.drop(drop_indices)
df_subset['Target'].value_counts()
dataTrain.loc[dataTrain['Target'] == 4] = df_subset
dataTrain = dataTrain.dropna()
dataTrain['Target'].value_counts().plot('bar')
XdataTrain = dataTrain.iloc[:, :-1]
YdataTrain = dataTrain.iloc[:, -1]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

accuracies = {}
melhor = 0
melhork=0
for k in range(1, 201):
    knn = KNeighborsClassifier(n_neighbors=k, p=1)
    knn.fit(XdataTrain,YdataTrain)
    scores = cross_val_score(knn, XdataTrain,YdataTrain, cv=10)
    if scores.mean()>melhor:
        melhor = scores.mean()
        melhork=k
melhork
knn = KNeighborsClassifier(n_neighbors=melhork, p=1)
knn.fit(XdataTrain,YdataTrain)
Ypredit = knn.predict(XdataTrain)
pd.Series(Ypredit).value_counts().plot('bar')
dataTest = pd.read_csv("../input/test.csv",
                      na_values = '?')
Ids = dataTest['Id']
dataTest = dataTest[['rooms', 'v18q1', 'r4h3', 'r4m3', 'r4t3', 'tamhog', 'cielorazo', 'abastaguano',
                       'sanitario1', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3',
                       'eviv1', 'eviv2', 'eviv3', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4',
                       'estadocivil5', 'estadocivil6', 'estadocivil7', 'hogar_nin', 'hogar_adul', 'hogar_mayor',
                       'meaneduc', 'computer', 'television', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3',
                       'lugar4', 'lugar5', 'lugar6', 'area1']]

dataTest = dataTest.fillna(0)

dataTest['epared1'].add(2 * dataTest['epared2'])
dataTest['epared1'].add(3 * dataTest['epared3'])
del dataTest['epared2']
del dataTest['epared3']

dataTest['etecho1'].add(2 * dataTest['etecho2'])
dataTest['etecho1'].add(3 * dataTest['etecho3'])
del dataTest['etecho2']
del dataTest['etecho3']

dataTest['eviv1'].add(2 * dataTest['eviv2'])
dataTest['eviv1'].add(3 * dataTest['eviv3'])
del dataTest['eviv2']
del dataTest['eviv3']

for i in range(2,8):
    dataTest['estadocivil1'].add(i * dataTest['estadocivil'+str(i)])
    del dataTest['estadocivil'+str(i)]

for i in range(2,7):
    dataTest['lugar1'].add(i * dataTest['lugar'+str(i)])
    del dataTest['lugar'+str(i)]

XdataTest = dataTest

knn = KNeighborsClassifier(n_neighbors=melhork, p=1)
knn.fit(XdataTrain,YdataTrain)
YdataPred1 = knn.predict(XdataTest)
pd.Series(YdataPred1).value_counts().plot('bar')
Ids = list(Ids)
YdataPred1 = list(YdataPred1)

submission1 = np.array([Ids, YdataPred1])
submission1 = pd.DataFrame(submission1.T,columns=['Id', 'Target'])
submission1['Target'] = submission1['Target'].astype(float).astype(int)
submission1
submission1.to_csv('res1.csv', index=False)