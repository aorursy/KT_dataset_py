# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/chronicKidney.csv")

print(data)
X= np.array(data.iloc[:,1:25].values)

Y=np.array(data.iloc[:,25].values)

print(Y)
val = [5,6,7,8,18,19,20,21,22,23]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

for v in val :

    X[:, v] = labelencoder_X.fit_transform(X[:, v])



labelencoder_y = LabelEncoder()

Y = labelencoder_y.fit_transform( Y)



print(X)

print(Y)

from sklearn.preprocessing import Imputer



miss= [1,2,3,4,9,10,11,12,13,14,15,16,17]

print(X[:2])

calcul=X[:,v ].reshape(-1,1)



imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 1:5])

z=np.array(X[:, 1:3])

X[:, 1:5] = imputer.transform(X[:, 1:5])

z1=np.array(X[:, 1:5])



imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 9:18])

z=np.array(X[:, 9:18])

X[:, 9:18] = imputer.transform(X[:, 9:18])

z1=np.array(X[:, 9:18])

print(X[:2])
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

X[:, [0]]= imp.fit_transform(X[:, [0]])
print(X)
np.unique(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state= 104,test_size=0.3)
from sklearn.svm import SVC
model_svc = SVC(C=0.1, gamma=10, max_iter=10000, class_weight="balanced")
model_svc.fit(X_train, y_train.ravel())
y_pred_regular_svc = model_svc.predict(X_test)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
acc_regular_svc = accuracy_score(y_test, y_pred_regular_svc)
print(acc_regular_svc)
y_pred_regular_svc 
prec_regular_svc = precision_score(y_test, y_pred_regular_svc)
print(prec_regular_svc)
recall_regular_svc = recall_score(y_test, y_pred_regular_svc)
print(recall_regular_svc)
classification_report(y_test, y_pred_regular_svc)
confusion_matrix(y_test, y_pred_regular_svc)
f1_regular_svc = f1_score(y_test, y_pred_regular_svc)
print(f1_regular_svc)
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train, y_train.ravel())
y_pred_regular_lgr = lgr.predict(X_test)
acc_regular_lgr = accuracy_score(y_test, y_pred_regular_lgr)
print(acc_regular_lgr)
prec_regular_lgr = precision_score(y_test, y_pred_regular_lgr)
print(prec_regular_lgr)
recall_regular_lgr = recall_score(y_test, y_pred_regular_lgr)
print(recall_regular_lgr)
classification_report(y_test, y_pred_regular_lgr)
confusion_matrix(y_test, y_pred_regular_lgr)
f1_regular_lgr = f1_score(y_test, y_pred_regular_lgr)
print(f1_regular_lgr)
from sklearn.linear_model import SGDClassifier
sgd_regular = SGDClassifier()
sgd_regular = sgd_regular.fit(X_train, y_train.ravel())
y_pred_regular_sgd = sgd_regular.predict(X_test)
acc_regular_sgd = accuracy_score(y_test, y_pred_regular_sgd)
print(acc_regular_sgd)
precision_regular_sgd = precision_score(y_test, y_pred_regular_sgd)
print(precision_regular_sgd)
recall_regular_sgd = recall_score(y_test, y_pred_regular_sgd)
print(recall_regular_sgd)
f1_regular_sgd = f1_score(y_test, y_pred_regular_sgd)
print(f1_regular_sgd)
classification_report(y_test, y_pred_regular_sgd)
confusion_matrix(y_test, y_pred_regular_sgd)
from sklearn.ensemble import RandomForestClassifier
rf_regular = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=104)
rf_regular.fit(X_train, y_train.ravel())
y_pred_regular_rf = rf_regular.predict(X_test)
acc_regular_rf = accuracy_score(y_test, y_pred_regular_rf)
print(acc_regular_rf)
from sklearn.linear_model import Perceptron
prc_regular = Perceptron()
prc_regular.fit(X_train, y_train.ravel())
y_pred_regular_prc = prc_regular.predict(X_test)
acc_regular_prc = accuracy_score(y_test, y_pred_regular_prc)
print(acc_regular_prc)
from sklearn.neural_network import MLPClassifier
mlprc_regular = MLPClassifier()
mlprc_regular.fit(X_train, y_train.ravel())
y_pred_regular_mlprc = mlprc_regular.predict(X_test)
acc_regular_mlprc = accuracy_score(y_test, y_pred_regular_mlprc)
print(acc_regular_mlprc)
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
from sklearn.ensemble import RandomForestClassifier
rfs = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=104)
rfs.fit(X_train_res, y_train_res.ravel())
y_pred_rfs = rfs.predict(X_test)
acc_rfs = accuracy_score(y_test, y_pred_rfs)
print(acc_rfs)
confusion_matrix(y_test, y_pred_rfs)
from sklearn.metrics import f1_score, recall_score
f1_rfs = f1_score(y_test, y_pred_rfs)
print(f1_rfs)
rfs_rec = recall_score(y_test, y_pred_rfs)
print(rfs_rec)
rfs_precision = precision_score(y_test, y_pred_rfs)
print(rfs_precision)
svms = SVC(C=0.1, gamma=100, kernel='linear')
svms.fit(X_train_res, y_train_res.ravel())
y_pred_svms = svms.predict(X_test)
acc_svms = accuracy_score(y_test, y_pred_svms)
print(acc_svms)
confusion_matrix(y_test, y_pred_svms)
X_train_res.shape
f1_svms = f1_score(y_test, y_pred_svms)
print(f1_svms)
precision_svms =precision_score(y_test, y_pred_svms)
print(precision_svms)
recall_svms =recall_score(y_test, y_pred_svms)
print(recall_svms)
classification_report(y_test, y_pred_svms)
confusion_matrix(y_test, y_pred_svms)
from sklearn.linear_model import LogisticRegression
lgs = LogisticRegression()
lgs.fit(X_train_res, y_train_res.ravel())
y_pred_lgs = lgs.predict(X_test)
acc_lgs = accuracy_score(y_test, y_pred_lgs)
print(acc_lgs)
confusion_matrix(y_test, y_pred_lgs)
classification_report(y_test, y_pred_lgs)
f1_lgs = f1_score(y_test, y_pred_lgs)
print(f1_lgs)
recall_lgs =recall_score(y_test, y_pred_lgs)
print(recall_lgs)
precision_lgs =precision_score(y_test, y_pred_lgs)
print(precision_lgs)
from sklearn.linear_model import SGDClassifier
sgds = SGDClassifier()
sgds.fit(X_train_res, y_train_res.ravel())
y_pred_sgds = sgds.predict(X_test)
acc_sgds = accuracy_score(y_test, y_pred_sgds)
print(acc_sgds)
confusion_matrix(y_test, y_pred_sgds)
classification_report(y_test, y_pred_sgds)
precision_sgds =precision_score(y_test, y_pred_sgds)
print(precision_sgds)
recall_sgds =recall_score(y_test, y_pred_sgds)
print(recall_sgds)
f1_sgds = f1_score(y_test, y_pred_sgds)
print(f1_sgds)
from sklearn.neural_network import MLPClassifier
mlps = MLPClassifier()
mlps.fit(X_train_res, y_train_res.ravel())
y_pred_mlps = mlps.predict(X_test)
acc_mlps = accuracy_score(y_test, y_pred_mlps)
print(acc_mlps)
confusion_matrix(y_test, y_pred_mlps)
classification_report(y_test, y_pred_mlps)
precision_mlps =precision_score(y_test, y_pred_mlps)
print(precision_mlps)
recall_mlps =recall_score(y_test, y_pred_mlps)

print(recall_mlps)
f1_mlps = f1_score(y_test, y_pred_mlps)

print(f1_mlps)
from imblearn.over_sampling import ADASYN



ada = ADASYN()



X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
rfa = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=104)
rfa.fit(X_train_res, y_train_res.ravel())
y_pred_rfa = rfa.predict(X_test)
acc_rfa = accuracy_score(y_test, y_pred_rfa)
print(acc_rfa)
f1_rfa =f1_score(y_test, y_pred_rfa)
print(f1_rfa)
confusion_matrix(y_test, y_pred_rfa)
classification_report(y_test, y_pred_rfa)
precision_rfa =precision_score(y_test, y_pred_rfa)
print(precision_rfa)
recall_rfa =recall_score(y_test, y_pred_rfa)
print(recall_rfa)
svca = SVC(C=0.1, gamma=100, kernel='linear')
svca.fit(X_train_res, y_train_res.ravel())
y_pred_svca = svca.predict(X_test)
acc_svca = accuracy_score(y_test, y_pred_svca)
print(acc_svca)
f1_svca = f1_score(y_test, y_pred_svca)
print(f1_svca)
confusion_matrix(y_test, y_pred_svca)
lra = LogisticRegression()
lra.fit(X_train_res, y_train_res.ravel())
y_pred_lra = lra.predict(X_test)
acc_lra = accuracy_score(y_test, y_pred_lra)
print(acc_lra)
classification_report(y_test, y_pred_lra)
confusion_matrix(y_test, y_pred_lra)
from sklearn.linear_model import SGDClassifier
sgda = SGDClassifier()
sgda.fit(X_train_res, y_train_res.ravel())
y_pred_sgda = sgda.predict(X_test)
acc_sgda = accuracy_score(y_test, y_pred_sgda)
print(acc_sgda)
confusion_matrix(y_test, y_pred_sgds)
classification_report(y_test, y_pred_sgds)
from sklearn.neural_network import MLPClassifier
mlpa = MLPClassifier()
mlpa.fit(X_train_res, y_train_res.ravel())
y_pred_mlpa = mlpa.predict(X_test)
acc_mlpa = accuracy_score(y_test, y_pred_mlpa)
print(acc_mlpa)
confusion_matrix(y_test, y_pred_mlpa)
classification_report(y_test, y_pred_mlpa)
prec_mlpa = precision_score(y_test, y_pred_mlpa)

print(prec_mlpa)
rec_mlpa = precision_score(y_test, y_pred_mlpa)

print(rec_mlpa)
f1_mlpa = precision_score(y_test, y_pred_mlpa)

print(f1_mlpa)
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()



X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
rfros = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=104)
rfros.fit(X_train_res, y_train_res.ravel())
y_pred_rfros = rfros.predict(X_test)
acc_rfros = accuracy_score(y_test, y_pred_rfros)
print(acc_rfros)
classification_report(y_test, y_pred_rfros)
confusion_matrix(y_test, y_pred_rfros)
f1_rfros = f1_score(y_test, y_pred_rfros)
print(f1_rfros)
precision_rfros =precision_score(y_test, y_pred_rfros)
print(precision_rfros)
recall_rfros =recall_score(y_test, y_pred_rfros)
print(recall_rfros)
svcros = SVC(C=0.1, gamma=100, kernel='linear')
svcros.fit(X_train_res, y_train_res.ravel())
y_pred_svcros = svcros.predict(X_test)
acc_svcros = accuracy_score(y_test, y_pred_svcros)
print(acc_svcros)
f1_svcros = f1_score(y_test, y_pred_svcros)
print(f1_svcros)
classification_report(y_test, y_pred_svcros)
confusion_matrix(y_test, y_pred_rfa)
precision_svcros =precision_score(y_test, y_pred_svcros)
print(precision_svcros)
recall_svcros =recall_score(y_test, y_pred_svcros)
print(recall_svcros)
lrros = LogisticRegression()
lrros.fit(X_train_res, y_train_res.ravel())
y_pred_lrros = lrros.predict(X_test)
acc_lrros = accuracy_score(y_test, y_pred_lrros)
print(acc_lrros)
classification_report(y_test, y_pred_lrros)
confusion_matrix(y_test, y_pred_lrros)
f1_lrros =f1_score(y_test, y_pred_lrros)
print(f1_lrros)
precision_lrros =precision_score(y_test, y_pred_lrros)
print(precision_lrros)
recall_lrros =recall_score(y_test, y_pred_lrros)
print(recall_lrros)
from sklearn.linear_model import SGDClassifier
sgdros = SGDClassifier()
sgdros.fit(X_train_res, y_train_res.ravel())
y_pred_sgdros = sgdros.predict(X_test)
acc_sgdros = accuracy_score(y_test, y_pred_sgdros)
print(acc_sgdros)
confusion_matrix(y_test, y_pred_sgdros)
classification_report(y_test, y_pred_sgdros)
f1_sgdros =f1_score(y_test, y_pred_sgdros)
print(f1_sgdros)
precision_sgdros =precision_score(y_test, y_pred_sgdros)
print(precision_sgdros)
recall_sgdros =recall_score(y_test, y_pred_sgdros)
print(recall_sgdros)
from sklearn.neural_network import MLPClassifier
mlpros = MLPClassifier()
mlpros.fit(X_train_res, y_train_res.ravel())
y_pred_mlpros = mlpros.predict(X_test)
acc_mlpros = accuracy_score(y_test, y_pred_mlpros)
print(acc_mlpros)
confusion_matrix(y_test, y_pred_mlpros)
classification_report(y_test, y_pred_mlpros)
f1_mlpros =f1_score(y_test, y_pred_mlpros)
print(f1_mlpros)
precision_mlpros =precision_score(y_test, y_pred_mlpros)
print(precision_mlpros)
recall_mlpros =recall_score(y_test, y_pred_mlpros)
print(recall_mlpros)