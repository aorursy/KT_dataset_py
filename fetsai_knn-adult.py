import pandas as pd
import sklearn
import matplotlib.pyplot as plt
adult = pd.read_csv("../input/adult-data/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

testAdult =pd.read_csv("../input/adult-data/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nadult=adult.dropna()

nadult.head()
nadult["age"].value_counts().plot(kind="bar")

plt.show()
nadult["education.num"].value_counts().plot(kind="pie")

plt.show()
nadult["hours.per.week"].value_counts().plot(kind="pie")

plt.show()
nadult["capital.loss"].value_counts()
nadult["capital.gain"].value_counts()
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult=nadult.income
nTestAdult=testAdult.dropna()
XtestAdult = nTestAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
i=0
s=0
c=1
n=0
while 1==1:
    knn = KNeighborsClassifier(n_neighbors=c)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    if scores.mean()>s:
        s=scores.mean()
        i=0
        n=c
    else:
        i+=1
        if i>10:
            break
    c+=1
n,s
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
prediction = pd.DataFrame(nTestAdult.Id)
prediction["income"] = YtestPred
prediction
prediction.to_csv("prediction.csv", index=False)