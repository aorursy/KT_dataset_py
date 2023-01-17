import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale,StandardScaler

from sklearn.impute import KNNImputer

from sklearn.svm import SVC

import lightgbm as lgb
#IMPORT TRAIN DATA

df_train=pd.read_csv("/kaggle/input/isteml2020/train.csv")

df_train.head()
#IMPORT TEST DATA

df_test=pd.read_csv("/kaggle/input/isteml2020/test.csv")

df_test.head()
Y_train = df_train['y']

Y_train = Y_train.to_numpy()



X_train = df_train.loc[:,df_train.columns!='y']

X_test = df_test.loc[:,df_test.columns!='y']
combo = pd.concat(objs=[X_train,X_test])

combo = pd.get_dummies(data=combo,columns=["x9","x16","x17","x18","x19"],dummy_na=True,drop_first=True)

X_train_dummy=pd.DataFrame(data=combo[0:Y_train.shape[0]])

X_test_dummy=pd.DataFrame(data=combo[Y_train.shape[0]:])
X_train_normalized = scale(X_train_dummy)

X_test_normalized=scale(X_test_dummy)



X_train_filled=KNNImputer(n_neighbors=10,weights='distance').fit_transform(X_train_normalized)

X_train_filled=pd.DataFrame(data=X_train_filled)

X_train_filled.columns=X_train_dummy.columns



X_test_filled=KNNImputer(n_neighbors=10,weights='distance').fit_transform(X_test_normalized)

X_test_filled=pd.DataFrame(data=X_test_filled)

X_test_filled.columns=X_test_dummy.columns
X_train = X_train_filled

X_test = X_test_filled



X_train_lgbm = X_train.to_numpy()

X_test_lgbm = X_test.to_numpy()
sc = StandardScaler()

X_train_lgbm = sc.fit_transform(X_train_lgbm)

X_test_lgbm = sc.transform(X_test_lgbm)
svmc=SVC(probability=True,random_state=0)

svmc.fit(X_train,Y_train)

predsvm=svmc.predict_proba(X_test)
d_train = lgb.Dataset(X_train_lgbm,label=Y_train)

params={}

params['learning_rate']=0.01

params['objective']='binary'

params['metric']='binary_logloss'

params['sub_feature']=0.65

params['num_leaves']=55

params['min_data']=10

params['max_bin']=730

params['random_state']=0



clf = lgb.train(params,d_train,2451)



predlgbm = clf.predict(X_test_lgbm)
pred = predsvm[:,1]

pred = (0.05*pred+1.3*predlgbm)/1.35
out=pd.DataFrame()

out["Id"]=range(0,4000)

out["Predicted"]=pd.DataFrame(pred)

out.head()
out.to_csv('results_out.csv', index=False)