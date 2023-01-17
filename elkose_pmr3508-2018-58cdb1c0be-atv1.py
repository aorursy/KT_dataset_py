import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
adult = pd.read_csv("../input/adultbasedata/train_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult = adult.drop(adult.index[0])
adult.shape
adult.head()
adult.describe()
adultgreat50k = adult[adult.Target != '<=50K']
adultless50k = adult[adult.Target != '>50K']
nadultgreat50k = adultgreat50k.dropna()
numadultgreat50k = nadultgreat50k.apply(preprocessing.LabelEncoder().fit_transform)
nadultless50k = adultless50k.dropna()
numadultlesst50k = nadultless50k.apply(preprocessing.LabelEncoder().fit_transform)
adultgreat50k.describe()
numadultgreat50k.describe()
adultless50k.describe()
numadultlesst50k.describe()
adultgreat50k["Workclass"].value_counts().plot(kind="bar")
adultless50k["Workclass"].value_counts().plot(kind="bar")
adultgreat50k["Race"].value_counts().plot(kind="bar")
adultgreat50k["Education"].value_counts().plot(kind="bar")
adultgreat50k["Education-Num"].value_counts().plot(kind="bar")
adultgreat50k["Martial Status"].value_counts().plot(kind="bar")
adultgreat50k["Sex"].value_counts().plot(kind="bar")
adultgreat50k["Relationship"].value_counts().plot(kind="bar")
adultgreat50k["Occupation"].value_counts().plot(kind="bar")
adultgreat50k["Age"].value_counts().plot(kind="pie")
adultgreat50k["Country"].value_counts().plot(kind="bar")
adultgreat50k["Capital Gain"].value_counts().plot(kind="pie")
adultgreat50k["Capital Loss"].value_counts().plot(kind="pie")
adultless50k["Capital Loss"].value_counts().plot(kind="bar")
nadult = adult.dropna()
testAdult = pd.read_csv("../input/adultbasedata/test_data.csv",
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"],
        sep=r'\s*,\s*',
        engine='python',
        na_values='.')
testAdult = testAdult.drop(testAdult.index[0])
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
XtestAdult = testAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
media = np.zeros(12)
for k in range(24,36):
    classficador = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(classficador, Xadult, Yadult, cv=10)
    media[k-24] = np.mean(score)
    
np.amax(media)
K = np.argmax(media)+24
K
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = testAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdultnum = numTestAdult.iloc[:,0:14]
Xadult = numAdult[["Age", "Workclass", "Education-Num","Martial Status",
        "Occupation", "Race", "Capital Gain", "Capital Loss",
        "Hours per week","Country"]]
XtestAdultnum = numTestAdult[["Age", "Workclass", "Education-Num","Martial Status",
        "Occupation", "Race", "Capital Gain", "Capital Loss",
        "Hours per week","Country"]]
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(Xadult,Yadult)
YtestPrednum = knn.predict(XtestAdultnum)
Result = [0 for i in range(len(YtestPrednum))]
for i in range (len(YtestPrednum)):
    if (YtestPrednum[i] == 0):
        Result[i] ="<=50K"
    else:
        Result[i] =">50K"

df = pd.DataFrame(testAdult.index)
df["Id"] = testAdult.index
df["income"] = Result
Send = df.drop(columns=[0])
Send.to_csv("teste.csv",index=False)