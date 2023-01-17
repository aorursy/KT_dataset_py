import pandas as pd
import sklearn
adult = pd.read_csv("../input/adult-db/train_data.csv",header=0, index_col=0, na_values="?")
print(adult.shape)
adult.head()
adult["native.country"].value_counts()
import matplotlib.pyplot as plt
%matplotlib inline
adult["age"].value_counts().plot(kind="bar")
adult["sex"].value_counts()
adult["education.num"].value_counts().plot(kind="bar")
adult["occupation"].value_counts().plot(kind="bar")
pd.unique(adult["relationship"])
from sklearn.preprocessing import LabelEncoder
from statistics import mode
import numpy as np
nadult = adult.dropna()
adult_fill = adult.fillna(-1)
for col in [1,3,5,6,7,8,9,13]:
    nadult = adult.iloc[:,col].dropna()
    label_encoder = LabelEncoder().fit(nadult)
    nadult_encoded = label_encoder.transform(nadult)
    adult_fill.iloc[np.where(adult_fill.iloc[:,col].values==-1)[0],col] = label_encoder.inverse_transform([int(mode(nadult_encoded))])
for col in [0,2,4,10,11,12]:
    adult_fill.iloc[np.where(adult_fill.iloc[:,col].values==-1)[0],col] = int(np.mean(adult.iloc[:,col].dropna().values))
adult_fill
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
col_inds = [0,1,4,5,6,7,8,10,11,12] # 0,1 [0,1,3,4,5,6,7,8,9,10,13] 2 [0,1,3,4,5,6,7,8,9,10,11,12,13] 3 [0,1,4,5,6,7,8,10,11,12]
Xadult_unscaled = adult_fill.iloc[:,col_inds].apply(LabelEncoder().fit_transform)
Xadult = minmaxscaler.fit_transform(Xadult_unscaled)
Yadult = adult_fill.income
print(Xadult_unscaled.columns.values)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
score_medio = np.zeros(50)
std_score = np.zeros(50)
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i, p=1)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10)
    score_medio[i-1]=np.mean(scores)
    std_score[i-1]=np.std(scores)
print(np.argmax(score_medio)+1)
print(np.amax(score_medio))
plt.errorbar(range(1,51), score_medio, yerr=1.96*np.array(std_score), fmt='-o')
testAdult = pd.read_csv("../input/adult-db/test_data.csv",header=0, index_col=0, na_values="?")
testAdult.shape
testAdult_fill = testAdult.fillna(-1)
for col in [1,3,5,6,7,8,9,13]:
    nTestAdult = testAdult.iloc[:,col].dropna()
    label_encoder = LabelEncoder().fit(nTestAdult)
    ntestAdult_encoded = label_encoder.transform(nTestAdult)
    testAdult_fill.iloc[np.where(testAdult_fill.iloc[:,col].values==-1)[0],col] = label_encoder.inverse_transform([int(mode(ntestAdult_encoded))])
for col in [0,2,4,10,11,12]:
    testAdult_fill.iloc[np.where(testAdult_fill.iloc[:,col].values==-1)[0],col] = int(np.mean(testAdult.iloc[:,col].dropna().values))
testAdult_fill
XtestAdult_unscaled = testAdult_fill.iloc[:,col_inds].apply(LabelEncoder().fit_transform)
XtestAdult = minmaxscaler.transform(XtestAdult_unscaled)
knn = KNeighborsClassifier(n_neighbors=34,p=1)
knn.fit(Xadult,Yadult)
YtestAdult = knn.predict(XtestAdult)
YtestAdult
prediction = pd.DataFrame(testAdult.index)
prediction["income"] = YtestAdult
prediction
prediction.to_csv("adult_prediction_5.csv", index=False)