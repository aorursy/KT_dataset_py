import pandas as pd

import numpy as np

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train = train.drop('id',1)

array = train.values

X = array[:,1:58]

Y = array[:,0]
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X, Y)

print(model.feature_importances_)
from sklearn.decomposition import PCA

# feature extraction

pca = PCA(n_components=20)

fit = pca.fit(X)

# summarize components

print("Explained Variance: %s")

print(fit.explained_variance_ratio_.cumsum())
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

rfe = RFE(model, 20)

fit = rfe.fit(X, Y)

print("Num Features: %d", fit.n_features_)

print("Selected Features: %s", fit.support_)

print("Feature Ranking: %s", fit.ranking_)
test.shape
feature_set1 = ['ps_ind_01','ps_ind_03','ps_ind_15','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_06_cat','ps_car_11_cat','ps_car_13','ps_car_14',

'ps_car_15',

'ps_calc_01',

'ps_calc_02',

'ps_calc_03',

'ps_calc_04',

'ps_calc_05',

'ps_calc_06']

feature_set2 = ['ps_ind_05_cat',

'ps_ind_06_bin',

'ps_ind_07_bin',

'ps_ind_08_bin',

'ps_ind_09_bin',

'ps_ind_10_bin',

'ps_ind_12_bin',

'ps_ind_16_bin',

'ps_ind_17_bin',

'ps_ind_18_bin',

'ps_reg_01',

'ps_reg_02',

'ps_reg_03','ps_car_03_cat','ps_car_07_cat','ps_car_10_cat','ps_car_11','ps_car_12','ps_car_13','ps_ind_15']

test_feature = ['target']
X1 = train[feature_set1]

X2 = train[feature_set2]

Y = train[test_feature]
from sklearn.model_selection import train_test_split

k = len(train.index)

k = int(k*0.7)

X_train = X1[:k]

X_test = X1[k:]

Y_train = Y[:k]

Y_test = Y[k:]
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression()

LogReg.fit(X_train, Y_train)

y_pred = LogReg.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy_score(Y_test, y_pred)
X_Train = train[feature_set1]

Y_Train = train[test_feature]

X_Test = test[feature_set1]

#Y_Test = test[test_feature]

X_Test.shape
LogReg = LogisticRegression()

LogReg.fit(X_Train, Y_Train)

#Y_Pred = LogReg.predict(X_Test)

Y_Pred = LogReg.predict_proba(X_Test)

#accuracy_score(Y_Test, Y_Pred)
df = pd.DataFrame(data=Y_Pred)

df = df.drop(df.columns[1], axis=1)



df = df.round(1)

df.to_csv('Output.csv',index = True)