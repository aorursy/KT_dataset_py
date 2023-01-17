# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
inp_train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
inp_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
inp_train.shape
inp_test.shape
inp_train.describe()
inp_train.isnull().any().sum()
inp_test.isnull().any().sum()
inp_train['label'].value_counts().sort_values(ascending=False)
tgt_col=inp_train['label']
inp_train.drop('label',inplace=True,axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(inp_train,tgt_col,test_size=0.3,random_state=2)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500,n_jobs=-1)
rfc.fit(X_train,y_train)
y_rfc_pred=rfc.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_rfc_pred)
cm
from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,y_rfc_pred)
print(classification_report(y_test,y_rfc_pred))
from xgboost import XGBClassifier
XGB_class=XGBClassifier(n_estimators=1000,n_jobs=-1)
XGB_class.fit(X_train,y_train)
y_xgb_pred=XGB_class.predict(X_test)
xgb_cm=confusion_matrix(y_test,y_xgb_pred)
print(xgb_cm)
accuracy_score(y_test,y_xgb_pred)
#XGB_class.fit(inp_train,tgt_col)
#y_sub_pred=XGB_class.predict(inp_test)
#submission_xgb = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series((y_sub_pred),name="Label")],axis = 1)
from sklearn.preprocessing import MinMaxScaler
MMS=MinMaxScaler()
inp_train_mm=pd.DataFrame(std_scaling.fit_transform(inp_train))
inp_test_mm=pd.DataFrame(std_scaling.transform(inp_test))
inp_train_mm.columns=inp_train.columns
inp_test_mm.columns=inp_test.columns
inp_train_mm.describe()
X_train_nn,X_test_nn,y_train_nn,y_test_nn=train_test_split(inp_train_mm,tgt_col)
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(100), max_iter=500, alpha=1e-4,

                    solver='sgd', verbose=10, random_state=5,

                    learning_rate_init=.01)
mlp.fit(X_train_nn,y_train_nn)
y_pred_nn=mlp.predict(X_test_nn)
cm_nn=confusion_matrix(y_test_nn,y_pred_nn)
print(cm_nn)
print(classification_report(y_test_nn,y_pred_nn))
accuracy_score(y_test_nn,y_pred_nn)
mlp_1=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

              beta_2=0.999, early_stopping=False, epsilon=1e-08,

              hidden_layer_sizes=100, learning_rate='constant',

              learning_rate_init=0.01, max_iter=500, momentum=0.9,

              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,

              random_state=5, shuffle=True, solver='sgd', tol=0.0001,

              validation_fraction=0.1, verbose=10, warm_start=False)
mlp_1.fit(X_train,y_train)
nn_whole=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,

              beta_2=0.999, early_stopping=False, epsilon=1e-08,

              hidden_layer_sizes=100, learning_rate='constant',

              learning_rate_init=0.01, max_iter=500, momentum=0.9,

              n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,

              random_state=5, shuffle=True, solver='sgd', tol=0.0001,

              validation_fraction=0.1, verbose=10, warm_start=False)
nn_whole.fit(inp_train_mm,tgt_col)
nn_pred_fin=nn_whole.predict(inp_test_mm)
nn_df=pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series((nn_pred_fin),name="Label")],axis = 1)
nn_df.to_csv('nn_sub.csv',index=False)