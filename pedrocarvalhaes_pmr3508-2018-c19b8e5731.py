import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
HHI = pd.read_csv("../input/train.csv", sep=r'\s*,\s*',
        engine='python',
        na_values="")
HHI
nHHI = HHI.drop(columns = "v2a1")
nHHI = nHHI.drop(columns ="v18q1")
nHHI = nHHI.drop(columns ="rez_esc")
nHHI = nHHI.drop(columns = "Id")
nHHI = nHHI.drop(columns ="idhogar")
nHHI = nHHI.drop(columns ="dependency")
nHHI = nHHI.drop(columns = "edjefe")
nHHI = nHHI.drop(columns ="edjefa")
nHHI = nHHI.drop(columns ="SQBmeaned")
nHHI = nHHI.drop(columns ="meaneduc")

nHHI.shape
nHHI.isnull().sum().sum()
testHHI = pd.read_csv("../input/test.csv", sep=r'\s*,\s*',
        engine='python',
        na_values="")
testHHI
testHHI.isnull().sum()
ntestHHI = testHHI.drop(columns = "v2a1")
ntestHHI = ntestHHI.drop(columns ="v18q1")
ntestHHI = ntestHHI.drop(columns ="rez_esc")
ntestHHI = ntestHHI.drop(columns ="Id")
ntestHHI = ntestHHI.drop(columns ="idhogar")
ntestHHI = ntestHHI.drop(columns ="dependency")
ntestHHI = ntestHHI.drop(columns ="edjefe")
ntestHHI = ntestHHI.drop(columns ="edjefa")
ntestHHI = ntestHHI.drop(columns ="SQBmeaned")
ntestHHI = ntestHHI.drop(columns ="meaneduc")

ntestHHI.shape
ntestHHI.isnull().sum().sum()
XHHI = nHHI.iloc[:,0:132]
XHHI.shape
YHHI = nHHI.Target
XtestHHI = ntestHHI.iloc[:,0:134]
XtestHHI.shape
k=1
v=[]
K=[]
while k<=200:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(XHHI, YHHI)
    scores = cross_val_score(knn, XHHI, YHHI, cv=10)
    x=np.mean(scores)
    print(x)
    v.append(x)
    K.append(k)
    k+=1
print(np.amax(v),np.argmax(v))
vetor = pd.DataFrame(data = v)

plt.plot(K, vetor)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(XHHI,YHHI)
scores = cross_val_score(knn, XHHI, YHHI, cv=10)
scores
np.mean(scores)
YtestHHIPred = knn.predict(XtestHHI)
YtestHHIPred
predicted = pd.DataFrame(data = YtestHHIPred)
predicted[0].value_counts()
predicted[0].value_counts().sort_index()
predicted[0].value_counts().sort_index().plot(kind = 'bar')

result = np.vstack((testHHI["Id"],YtestHHIPred)).T
x = ['Id','Target']
Result = pd.DataFrame(columns=x ,data = result)
Result
Result.to_csv('resultados_Costa_Rica.csv', index=False)