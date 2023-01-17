import pandas as pd
import sklearn
cr = pd.read_csv('../input/train.csv',header=0, index_col=0, na_values=[''])
#cr = cr_p.loc[cr_p['parentesco1'] == 0]
cr.head()
print(cr.shape)
cr['parentesco1'].value_counts()
cr = cr
cr.shape
import matplotlib.pyplot as plt
%matplotlib inline
print(cr.select_dtypes(include=object).head(0).columns)
from sklearn.preprocessing import LabelEncoder
from statistics import mode
import numpy as np
cr_fill = cr.fillna(-1)
for col in range(1,cr.select_dtypes(include=object).shape[1]):
    ncr = cr.select_dtypes(include=object).iloc[:,col]
    colname = ncr.head(0).name
    cr_fill.loc[cr_fill[colname]=='yes',colname]='1'
    cr_fill.loc[cr_fill[colname]=='no',colname]='0'
    cr_fill[colname] = pd.to_numeric(cr_fill[colname])
for col in range(cr.select_dtypes(exclude=object).shape[1]):
    ncr = cr.select_dtypes(exclude=object).iloc[:,col].dropna()
    colname = ncr.head(0).name
    cr_fill.loc[cr_fill[colname]==-1,colname] = np.mean(ncr.values)
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()
households = np.unique(cr_fill['idhogar'].values)
cr_par = cr_fill.loc[cr['parentesco1']==1].iloc[:,:-10]
for household in households:
    cr_par.loc[cr_par['idhogar']==household,['parentesco'+str(i) for i in range(1,13)]] = cr_fill.loc[cr_fill['idhogar']==household,['parentesco'+str(i) for i in range(1,13)]].sum().values
Xcr_unscaled = cr_par.select_dtypes(exclude=object) #.apply(LabelEncoder().fit_transform)
Xcr = minmaxscaler.fit_transform(Xcr_unscaled)
Ycr = cr_fill.loc[cr['parentesco1']==1].Target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
score_medio = np.zeros(100)
std_score = np.zeros(100)
for i in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=i, p=1)
    scores = cross_val_score(knn, Xcr, Ycr, cv=10)
    score_medio[i-1]=np.mean(scores)
    std_score[i-1]=np.std(scores)
print(np.argmax(score_medio)+1)
print(np.amax(score_medio))
plt.errorbar(range(1,101), score_medio, yerr=1.96*np.array(std_score), fmt='-o')
testCr = pd.read_csv('../input/test.csv',header=0, index_col=0, na_values="?")
testCr.shape
testCr_fill = testCr.fillna(-1)
for col in range(1,testCr.select_dtypes(include=object).shape[1]):
    ncr = testCr.select_dtypes(include=object).iloc[:,col]
    colname = ncr.head(0).name
    testCr_fill.loc[testCr_fill[colname]=='yes',colname]='1'
    testCr_fill.loc[testCr_fill[colname]=='no',colname]='0'
    testCr_fill[colname] = pd.to_numeric(testCr_fill[colname])
for col in range(testCr.select_dtypes(exclude=object).shape[1]):
    ncr = testCr.select_dtypes(exclude=object).iloc[:,col].dropna()
    colname = ncr.head(0).name
    testCr_fill.loc[testCr_fill[colname]==-1,colname] = np.mean(ncr.values)
households = np.unique(testCr_fill['idhogar'].values)
testCr_par = testCr_fill.loc[testCr_fill['parentesco1']==1].copy()
for household in households:
    testCr_fill.loc[testCr_fill['idhogar']==household,['parentesco'+str(i) for i in range(1,13)]] = testCr_fill.loc[testCr_fill['idhogar']==household,['parentesco'+str(i) for i in range(1,13)]].sum().values
print(testCr_fill.loc[testCr_fill['idhogar']==household,['parentesco'+str(i) for i in range(1,13)]])
XtestCr_unscaled = testCr_fill.select_dtypes(exclude=object).iloc[:,:-9] #.apply(LabelEncoder().fit_transform)
XtestCr = minmaxscaler.transform(XtestCr_unscaled)
XtestCr.shape
knn = KNeighborsClassifier(n_neighbors=30,p=1)
knn.fit(Xcr,Ycr.astype('int32'))
YtestCr = knn.predict(XtestCr)
YtestCr.shape
prediction = pd.DataFrame(testCr.index)
prediction["Target"] = YtestCr
prediction
prediction.to_csv("cr_prediction.csv", index=False)
