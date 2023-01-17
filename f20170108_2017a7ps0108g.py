import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import mean_squared_error as MSE

from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.tree import DecisionTreeRegressor as DTR
df = pd.read_csv("/kaggle/input/bits-f464-l1/train.csv")

df_test = pd.read_csv("/kaggle/input/bits-f464-l1/test.csv")

df_sam = pd.read_csv("/kaggle/input/bits-f464-l1/sampleSubmission.csv")
aId = 1*df.a0 + 2*df.a1 + 3*df.a2 + 4*df.a3 + 5*df.a4 + 6*df.a5 + 7*df.a6

tId = 1*df_test.a0 + 2*df_test.a1 + 3*df_test.a2 + 4*df_test.a3 + 5*df_test.a4 + 6*df_test.a5 + 7*df_test.a6
### find sepearately

err = 0

for ag in range(1,8):

  df_new = df[aId==ag].drop(columns=['id','time','a0','a1','a2','a3','a4','a5','a6'])



  X = df_new.iloc[:,:-1]

  y = df_new["label"]



  splt = (int)(X.shape[0]*0.75)

  X_train = X[:splt]

  X_test = X[splt:]

  y_train = y[:splt]

  y_test = y[splt:]

  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



  clf = RFR(n_estimators=150, n_jobs=-1 )#, min_samples_split=8, min_samples_leaf=3)

  # clf = DTR()

  clf_mod = clf.fit(X_train,y_train)



  pred = clf_mod.predict(X_test)

  temp_err = MSE(y_test,pred,squared=False)

  err+=temp_err*temp_err

  print(temp_err)



  df_new = df_test[tId==ag].drop(columns=['id','time','a0','a1','a2','a3','a4','a5','a6'])

  X = df_new

  pred = clf_mod.predict(X)



  for i in range(X.shape[0]):

    df_sam.label[7*i+ag-1]=pred[i]

print(np.sqrt(err/7))
###all agents combined

drop_list = np.where(np.array((df.iloc[:,:-8].nunique(dropna=False)<5))==True)[0]

print(drop_list)

df_new = df.drop(df.columns[drop_list],axis=1)

df_new = df_new.drop(columns=['id','time'])

X = df_new.iloc[:,:-1]

y = df_new["label"]



splt = (int)(X.shape[0]*0.75)

X_train = X[:splt]

X_test = X[splt:]

y_train = y[:splt]

y_test = y[splt:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



clf = RFR(n_estimators=150, n_jobs=-1 )#, min_samples_split=5, min_samples_leaf=3)

clf_mod = clf.fit(X_train,y_train)

pred = clf_mod.predict(X_test)

temp_err = MSE(y_test,pred,squared=False)

print(temp_err)



df_new = df_test.drop(df.columns[drop_list],axis=1)

df_new = df_new.drop(columns=['id','time'])

X = df_new

pred = clf_mod.predict(X)

df_sam['label'] = pred
df_sam.to_csv("pred.csv",index='False')