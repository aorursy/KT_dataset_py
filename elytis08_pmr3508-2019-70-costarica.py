import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn import preprocessing
import os

print(os.listdir("../input"))
#HHI = pd.read_csv("../input/train.csv", sep=r'\s*,\s*', engine='python', na_values="")

HHI = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv",sep=r'\s*,\s*', engine='python', na_values="?")

HHItest = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv",sep=r'\s*,\s*', engine='python', na_values="?")
print(HHI.shape)

print(HHItest.shape)
HHI
missingDatas = pd.DataFrame(HHI.isnull().sum()).rename(columns = {0: 'totNull'})

missingDatas.sort_values('totNull', ascending = False).head(8)
missingDatasTest = pd.DataFrame(HHItest.isnull().sum()).rename(columns = {0: 'totNull'})

missingDatasTest.sort_values('totNull', ascending = False).head(8)
HHI_old = HHI

HHItest_old = HHItest



dropList = ["rez_esc","v18q1","v2a1"]

for col in dropList:

    HHI = HHI.drop(columns = col)

    HHItest = HHItest.drop(columns = col)



HHI.dropna(axis=0, how='any', inplace=True)

HHItest.dropna(axis=0, how='any', inplace=True)

    

    

print(HHI.shape)

print(HHItest.shape)
missingDatas = pd.DataFrame(HHI.isnull().sum()).rename(columns = {0: 'totNull'})

missingDatas.sort_values('totNull', ascending = False).head(8)
print(HHI.info())

print(" ")

print(HHItest.info())
print(HHI.select_dtypes('object').head())

print(" ")

print(HHItest.select_dtypes('object').head())
dropList = ["Id","idhogar"]

IdDF = pd.DataFrame(HHI["Id"])

IdDFtest = pd.DataFrame(HHItest["Id"])



for col in dropList:

    HHI = HHI.drop(columns = col)

    HHItest = HHItest.drop(columns = col)





mapping = {"yes": 1, "no": 0}



# Apply same operation to both train and test

for df in [HHI, HHItest]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)
print(HHI.info())

print(" ")

print(HHItest.info())
XHHI = HHI.drop(columns = "Target")

YHHI = HHI["Target"]
knn = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(knn, XHHI, YHHI, cv=8)

print(scores)

print(scores.mean())
scoresMeanList=[]

kList=[]

for k in range(1,101):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, XHHI, YHHI, cv=8)

    scoresMeanList.append(scores.mean())

    kList.append(k)    
maxi = max(scoresMeanList)

print("The best amount of neighbors for this forst test is {}, with a score of {}".format(scoresMeanList.index(maxi),maxi))

plt.plot(kList,scoresMeanList)
knn = KNeighborsClassifier(n_neighbors=73)

knn.fit(XHHI,YHHI)

YHHIpred = knn.predict(HHItest)

YHHIpred = pd.DataFrame(data = YHHIpred)
print(YHHIpred[0].value_counts())

YHHIpred[0].value_counts().plot("bar")
YHHI = pd.DataFrame(data = YHHI)

print(YHHI["Target"].value_counts())

YHHI["Target"].value_counts().plot("bar")
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(XHHI,YHHI)

YHHIpred = knn.predict(HHItest)

YHHIpred = pd.DataFrame(data = YHHIpred)



print(YHHIpred[0].value_counts())

YHHIpred[0].value_counts().plot("bar")
results =[]

Ids = IdDFtest.values

target = YHHIpred.values

for i in range(len(Ids)):

    results.append([Ids[i][0],target[i][0]])

results
results = pd.DataFrame(columns=["Id","Target"] ,data = results)

results.head(21)
results.to_csv('results.csv', index=False)