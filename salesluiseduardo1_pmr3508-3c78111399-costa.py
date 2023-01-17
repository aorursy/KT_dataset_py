import pandas as pd
import sklearn
import os
costa_rican = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

costa_rican
costa_rican.shape
costa_rican["Target"].value_counts() #maior parte é não vulnerable
costa_rican["v2a1"].value_counts().plot(kind="bar")
ncosta_rican=costa_rican.dropna()
ncosta_rican
testcosta_rican = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
ntestcosta_rican= testcosta_rican.dropna()
ntestcosta_rican.shape
testcosta_rican
Xcosta_rican=ncosta_rican.iloc[:,1:50]
Xcosta_rican
Xcosta_rican.shape
Ycosta_rican = ncosta_rican.Target
Ycosta_rican
Ycosta_rican.shape
Xtestcosta_rican=ntestcosta_rican.iloc[:,1:50]
Xtestcosta_rican
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xcosta_rican, Ycosta_rican, cv=5)
scores
knn.fit(Xcosta_rican, Ycosta_rican)
YtestPred = knn.predict(Xtestcosta_rican)
YtestPred.shape
YtestReal = pd.read_csv("../input/costa-rican-household-poverty-prediction/sample_submission.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
YtestReal
YtestReal=YtestReal.dropna()
from sklearn.metrics import accuracy_score
