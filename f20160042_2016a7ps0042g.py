import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")
train.fillna(value=train.mean(),inplace=True)
test = test.fillna(value=test.mean())
train = train.drop(['id'],axis=1)

test = test.drop(['id'],axis=1)
y=train['rating']

X=train.drop(['rating'],axis=1)

X.head()
X=pd.get_dummies(X,columns=['type'])

X=X.drop(['type_old'],axis=1)
test=pd.get_dummies(test,columns=['type'])

test=test.drop(['type_old'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=1200, random_state = 69, n_jobs=-1)

regressor.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error



y_pred_val = regressor.predict(X_val)

y_pred_val = np.rint(y_pred_val)

np.sqrt(mean_squared_error(y_val,y_pred_val))
regressor.fit(X,y)

y_pred = regressor.predict(test)

y_pred = np.rint(y_pred)
final_test=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

X_test_id = final_test['id']

final = pd.concat([X_test_id,pd.DataFrame(y_pred)],axis=1)

final.columns=['id','rating']

final.to_csv('final2.csv',index=False)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math



train = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

test = pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")



train.fillna(value=train.mean(),inplace=True)



test = test.fillna(value=test.mean())



train = train.drop(['id'],axis=1)

test = test.drop(['id'],axis=1)



y=train['rating']

X=train.drop(['rating'],axis=1)

X.head()



X=pd.get_dummies(X,columns=['type'])



test=pd.get_dummies(test,columns=['type'])



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)



from sklearn.ensemble import ExtraTreesRegressor



regressor = ExtraTreesRegressor(n_estimators=1000, random_state = 42)

regressor.fit(X_train, y_train)



from sklearn.metrics import mean_squared_error



y_pred_val = regressor.predict(X_val)

y_pred_val = np.rint(y_pred_val)

np.sqrt(mean_squared_error(y_val,y_pred_val))



regressor.fit(X,y)

y_pred = regressor.predict(test)

y_pred = np.rint(y_pred)



final_test=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/test.csv")

X_test_id = final_test['id']

final = pd.concat([X_test_id,pd.DataFrame(y_pred)],axis=1)

final.columns=['id','rating']

final.to_csv('final1.csv',index=False)