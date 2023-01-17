from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv("../input/train.csv")
df.head()
X = df.iloc[:,1:]

y = df.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
# instiancing the RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100)

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
rfc.score(X_test,y_test)
rfc.score(X_train,y_train)
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(rfc,X,y,cv= 5)
cv_score
np.mean(cv_score)