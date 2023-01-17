import os
print((os.listdir('../input/')))
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('../input/web-club-recruitment-2018/train.csv')
df_test = pd.read_csv('../input/web-club-recruitment-2018/test.csv')




X = df_train.loc[:, 'X1':'X23']

y = df_train.loc[:, 'Y']

rf = RandomForestClassifier(n_estimators=105,max_features=None,random_state=133,min_samples_leaf=57,max_depth=7,min_samples_split=3)
##X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)


from sklearn.model_selection import cross_val_score,KFold
k_fold=KFold(n_splits=10)
scores=cross_val_score(rf,X,y,cv=k_fold)
print(scores)
print(scores.mean())

rf.fit(X, y)



test = df_test.loc[:, 'X1':'X23']

pred = rf.predict_proba(test)
result = pd.DataFrame(pred[:,1])
result.index.name = 'id'
result.columns = ['predicted_val']
result.to_csv('output.csv', index=True)
