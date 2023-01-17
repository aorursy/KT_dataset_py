import pandas as pd 
import sklearn
import os
print(os.listdir('../input'))
adult = pd.read_csv("../input/dataadult/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.head()
nadult = adult.dropna()
nadult
Xadult = nadult [["age","education.num","capital.gain","capital.loss","hours.per.week"]]
Yadult = nadult.income
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors= 5)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
testadult = pd.read_csv("../input/dataadult/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
Xtestadult = testadult[["age","education.num","capital.gain","capital.loss","hours.per.week"]]
YtestPred = knn.predict(Xtestadult)
YtestPred
arr1= testadult.iloc[:,0].values
arr1 = arr1.ravel()
dataset = pd.DataFrame({'Id':arr1[:],'income':YtestPred[:]})
dataset.to_csv("Adultscompetition.csv", index = False)
