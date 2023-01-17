import pandas as pd
import sklearn
costa = pd.read_csv("../input/costadataset/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
costa = costa.drop(columns=['v2a1','rez_esc','v18q1','dependency','edjefe','edjefa'])
ncosta = costa.dropna()
ncosta
testCosta = pd.read_csv("../input/costadataset/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
Xcosta = ncosta[['hacdor','rooms','hacapo','v14a','refrig','tamviv','escolari',
                'SQBhogar_total','instlevel1','instlevel9','bedrooms','SQBovercrowding','tipovivi1',
                'tipovivi4','computer','television','qmobilephone','area1','area2','SQBhogar_nin','tamviv']]
Ycosta = ncosta.Target
XtestCosta = testCosta[['hacdor','rooms','hacapo','v14a','refrig','tamviv','escolari',
                'SQBhogar_total','instlevel1','instlevel9','bedrooms','SQBovercrowding','tipovivi1',
                'tipovivi4','computer','television','qmobilephone','area1','area2','SQBhogar_nin','tamviv']]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xcosta, Ycosta, cv=40)
scores
scores.mean()
knn.fit(Xcosta,Ycosta)
YtestPred = knn.predict(XtestCosta)
YtestPred
import numpy as np
result = np.vstack((testCosta["Id"], YtestPred)).T
x = ["Id","Target"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("resultados_costa.csv", index = False)
Resultado
