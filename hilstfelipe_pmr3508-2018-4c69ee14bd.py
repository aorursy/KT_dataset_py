import pandas as pd
import numpy as np
import sklearn 
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
crtrain = pd.read_csv(r'''../input/train.csv''',
          sep = r'\s*,\s*',
          engine = 'python',
          na_values = 'NaN')
crtrain.shape
crtest = pd.read_csv(r'''../input/test.csv''',
          sep = r'\s*,\s*',
          engine = 'python',
          na_values = 'NaN')
crtest.shape
newtrain = crtrain[:]
newtest = crtest[:]
for i in range(len(crtrain["Id"])):
    if crtrain.parentesco1[i] != 1:
        newtrain = newtrain.drop(i)
for i in range(len(crtest["Id"])):
    if crtest.parentesco1[i] != 1:
        newtest = newtest.drop(i)
newtrain.shape
newtest.shape
newtrain.v2a1 = newtrain.v2a1.fillna(value = 0, downcast=None)
newtest.v2a1 = newtest.v2a1.fillna(value = 0, downcast=None)
newtrain.v18q1 = newtrain.v18q1.fillna(value = 0, downcast=None)
newtest.v18q1 = newtest.v18q1.fillna(value = 0, downcast=None)
newtrain.qmobilephone = newtrain.qmobilephone.fillna(value = 0, downcast=None)
newtest.qmobilephone = newtest.qmobilephone.fillna(value = 0, downcast=None)
newtrain["epared"] = 2*newtrain["epared3"] + newtrain["epared2"]
newtrain["etecho"] = 2*newtrain["etecho3"] + newtrain["etecho2"]
newtrain["eviv"] = 2*newtrain["eviv3"] + newtrain["eviv2"]
newtest["epared"] = 2*newtest["epared3"] + newtest["epared2"]
newtest["etecho"] = 2*newtest["etecho3"] + newtest["etecho2"]
newtest["eviv"] = 2*newtest["eviv3"] + newtest["eviv2"]
Xtrain = newtrain[["v2a1","v18q1","hhsize","noelec","sanitario1","energcocinar1","elimbasu1","epared","etecho","eviv","area1","qmobilephone"]]
Ytrain = newtrain.Target
means =[]
for num in range(1,31):
    knn = KNeighborsClassifier(n_neighbors = num)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    mean = statistics.mean(scores)
    means.append(mean) 
bestn = means.index(max(means))+1
bestn

knn = KNeighborsClassifier(n_neighbors = bestn)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
statistics.mean(scores)
statistics.pstdev(scores)
def percent(colum):
    return colum*100//float(sum(colum))
import matplotlib.pyplot as plt
# hhsize
newtrain["hhsize"].value_counts().plot(kind='bar')
pd.crosstab(newtrain["Target"],newtrain["hhsize"])
pd.crosstab(newtrain["hhsize"],newtrain["Target"]).plot()
# v18q1
newtrain["v18q1"].value_counts().plot(kind='bar')
targetxv18q1 = pd.crosstab(newtrain["v18q1"],newtrain["Target"])
targetxv18q1
# noelec
targetxnoelec = pd.crosstab(newtrain["noelec"],newtrain["Target"])
targetxnoelec
targetxnoelec.apply(percent,axis=1).plot()
# area1
targetxarea1 = pd.crosstab(newtrain["area1"],newtrain["Target"])
targetxarea1
targetxarea1.apply(percent,axis=1).plot()
# qmobilephone
newtrain["qmobilephone"].value_counts().plot(kind='bar')
pd.crosstab(newtrain["Target"],newtrain["qmobilephone"])
pd.crosstab(newtrain["qmobilephone"],newtrain["Target"]).plot()
# eviv
pd.crosstab(newtrain["eviv"],newtrain["Target"]).plot()
# etecho
pd.crosstab(newtrain["etecho"],newtrain["Target"]).plot()
# epared
pd.crosstab(newtrain["epared"],newtrain["Target"]).plot()
# sanitario1
pd.crosstab(newtrain["sanitario1"],newtrain["Target"]).plot()
# energcocinar1
pd.crosstab(newtrain["energcocinar1"],newtrain["Target"]).plot()
# elimbasu1
pd.crosstab(newtrain["elimbasu1"],newtrain["Target"]).plot()
Xtrain = newtrain[["v2a1","v18q1","hhsize","noelec","area1","qmobilephone","elimbasu1","eviv"]]
Xtest = newtest[["v2a1","v18q1","hhsize","noelec","area1","qmobilephone","elimbasu1","eviv"]]
knn = KNeighborsClassifier(n_neighbors = bestn)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
statistics.mean(scores)
knn.fit(Xtrain, Ytrain)
Ypred = knn.predict(Xtest)
newtest["sub"] = Ypred
# join to left
crtest = crtest.merge(newtest, how='left', left_on='idhogar', right_on='idhogar')
crtest.head()
submission = np.array([crtest['Id_x'], crtest['sub']])
submission
submit = pd.DataFrame(submission.T,
                     columns = ['Id','Target'])
submit['Target'] = submit['Target'].fillna(4).astype(int)
submit.Target.value_counts().plot(kind = 'bar')
submit.to_csv("submit.csv",
             index=False)
