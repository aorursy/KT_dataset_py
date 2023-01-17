import numpy as np

import pandas as pd

import xgboost as xgb

import sklearn.model_selection

import sklearn.metrics
data = pd.read_csv('../input/creditcard.csv')
data.head()
df = data.drop('Time',axis=1)
X, Xt, Y, Yt = sklearn.model_selection.train_test_split(df.drop('Class', axis=1), df['Class'], test_size=0.20, random_state=10)
X.head()
Y.head()
model = xgb.XGBClassifier()
model.fit(X, Y)
Yp = model.predict(Xt)
CM = np.array(sklearn.metrics.confusion_matrix(Yp, Yt))

print(CM)
TPR = 1.0*CM[1,1]/(CM[1,0]+CM[1,1])

print(TPR)
FNR = 1.0*CM[0,1]/(CM[0,1]+CM[0,0])

print(FNR)